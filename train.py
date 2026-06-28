import os
import argparse
import logging
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 on Cloud Run with QLoRA")
    
    # Model and dataset arguments
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="google/gemma-4-E4B-it", 
        help="Hugging Face model ID to fine-tune (e.g., google/gemma-4-E4B-it or google/gemma-4-E4B)"
    )
    parser.add_argument(
        "--dataset_name_or_path", 
        type=str, 
        default="trl-lib/Capybara", 
        help="Hugging Face dataset name, or local path to JSONL/JSON dataset"
    )
    parser.add_argument(
        "--dataset_split", 
        type=str, 
        default="train", 
        help="Dataset split to use for training"
    )
    
    # LoRA (PEFT) Hyperparameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Training Hyperparameters
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory where local checkpoints are saved")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps")
    
    # Google Cloud Storage Integration
    parser.add_argument(
        "--gcs_bucket", 
        type=str, 
        default=None, 
        help="Optional: GCS bucket name (without gs://) to upload final adapter weights"
    )
    parser.add_argument(
        "--gcs_prefix", 
        type=str, 
        default="gemma-4-adapters", 
        help="GCS prefix directory under the bucket to store weights"
    )
    
    # Hugging Face Hub Integration
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Optional: Hugging Face repository ID (e.g. 'username/gemma-4-sentiment-adapter') to push fine-tuned adapter weights"
    )
    
    return parser.parse_args()

def upload_directory_to_gcs(local_dir, bucket_name, gcs_prefix):
    """Uploads all files in a local directory to a GCS bucket."""
    try:
        from google.cloud import storage
        logger.info(f"Uploading fine-tuned adapter weights from {local_dir} to gs://{bucket_name}/{gcs_prefix}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Compute relative path to maintain directory structure
                rel_path = os.path.relpath(local_path, local_dir)
                gcs_path = os.path.join(gcs_prefix, rel_path)
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded: {local_path} -> gs://{bucket_name}/{gcs_path}")
                
        logger.info("GCS upload complete!")
    except ImportError:
        logger.error("google-cloud-storage not installed. Cannot upload to GCS.")
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
def custom_prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None, compute_dtype=torch.bfloat16):
    """Memory-efficient model preparation for k-bit training on Gemma models.
    Only upcasts layernorms (norm, ln) to float32 to prevent upcasting embeddings
    and lm_head, saving ~8.75 GiB of VRAM from being allocated.
    Also handles downcasting residual bfloat16 buffers on FP16 hardware (like T4).
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if ("norm" in name or "ln" in name) and param.__class__.__name__ != "Params4bit":
            if param.dtype in [torch.float16, torch.bfloat16]:
                param.data = param.data.to(torch.float32)
    
    # If training on standard non-bf16 hardware (like T4), we must downcast any residual bfloat16 params/buffers to float16
    if compute_dtype == torch.float16:
        logger.info("Downcasting residual bfloat16 parameters and buffers to float16 for T4 compatibility...")
        for name, buf in model.named_buffers():
            if buf.dtype == torch.bfloat16:
                buf.data = buf.data.to(torch.float16)
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16 and param.__class__.__name__ != "Params4bit":
                param.data = param.data.to(torch.float16)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model

def resolve_hf_token():
    """Resolves Hugging Face Token from environment or Google Secret Manager."""
    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
        return os.environ["HF_TOKEN"]
    
    # Check if we can load from GCP Secrets
    try:
        from google.cloud import secretmanager
        project_id = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID")
        secret_name = os.environ.get("HF_TOKEN_SECRET_NAME", "HF_TOKEN")
        if project_id:
            logger.info(f"HF_TOKEN not in env. Attempting to fetch secret '{secret_name}' from GCP Secret Manager for project '{project_id}'...")
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            token = response.payload.data.decode("UTF-8").strip()
            logger.info("Successfully retrieved HF_TOKEN from GCP Secret Manager.")
            os.environ["HF_TOKEN"] = token
            return token
    except Exception as e:
        logger.debug(f"Could not retrieve secret from Secret Manager: {e}")
        
    return None


def main():
    args = parse_args()
    
    # Resolve HF Token
    token = resolve_hf_token()
    if not token:
        logger.warning(
            "HF_TOKEN environment variable is not set and could not be fetched from Secret Manager. "
            "Gemma 4 is a gated model, so you must provide an authorized token to download it."
        )
    
    # 1. Load Dataset
    logger.info(f"Loading dataset: {args.dataset_name_or_path}")
    dataset_path = args.dataset_name_or_path
    temp_local_file = None
    
    if dataset_path.startswith("gs://"):
        import tempfile
        import urllib.parse
        from google.cloud import storage
        
        parsed = urllib.parse.urlparse(dataset_path)
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip("/")
        
        # Keep original extension
        _, ext = os.path.splitext(blob_path)
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_local_file = temp_file.name
        temp_file.close()
        
        logger.info(f"Downloading dataset from GCS: {dataset_path} -> {temp_local_file}")
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(temp_local_file)
            dataset_path = temp_local_file
        except Exception as e:
            logger.error(f"Failed to download dataset from GCS: {e}")
            if os.path.exists(temp_local_file):
                os.unlink(temp_local_file)
            raise e
            
    if os.path.exists(dataset_path):
        # Determine format based on extension
        if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_path, split=args.dataset_split)
        elif dataset_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_path, split=args.dataset_split)
        else:
            raise ValueError("Unsupported local/downloaded dataset format. Must be JSON, JSONL, or CSV.")
        
        # Clean up temporary dataset file if downloaded from GCS
        if temp_local_file and os.path.exists(temp_local_file):
            logger.info(f"Cleaning up temporary dataset file: {temp_local_file}")
            os.unlink(temp_local_file)
    else:
        # Load from HF Hub
        dataset = load_dataset(dataset_path, split=args.dataset_split)
    
    logger.info(f"Dataset loaded. Number of examples: {len(dataset)}")
    
    # 2. Configure 4-bit quantization (QLoRA) based on GPU capabilities
    logger.info("Configuring 4-bit quantization (BitsAndBytes)...")
    
    # Check GPU capability for BF16 support (e.g., L4, A100, H100 support BF16, T4 does not)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        fp16 = False
        bf16 = True
        logger.info("GPU supports bfloat16. Setting compute_dtype=torch.bfloat16, bf16=True.")
    else:
        compute_dtype = torch.float16
        fp16 = True
        bf16 = False
        logger.info("bfloat16 is NOT supported or no GPU is available. Setting compute_dtype=torch.float16, fp16=True.")
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # 3. Load Base Model and Tokenizer
    logger.info(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Unwrap Gemma4ClippableLinear modules if present to prevent PEFT/LoRA errors
    unwrapped_count = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "Gemma4ClippableLinear":
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.linear)
            unwrapped_count += 1
    if unwrapped_count > 0:
        logger.info(f"Successfully unwrapped {unwrapped_count} Gemma4ClippableLinear modules.")
    
    # Prepare model for k-bit training (e.g. gradient checkpointing setup)
    model = custom_prepare_model_for_kbit_training(model, compute_dtype=compute_dtype)
    
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Recommended side for training CausalLM
    
    # Set standard Gemma chat template if not present (essential for training base models on conversational datasets)
    if tokenizer.chat_template is None:
        logger.info("Setting standard Gemma chat template on tokenizer.")
        tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
            "{% elif message['role'] == 'model' %}"
            "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<start_of_turn>model\n' }}"
            "{% endif %}"
        )
    
    # 4. LoRA Adapter Configuration
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 5. SFTTrainer Configuration
    logger.info("Initializing SFTTrainer...")
    
    # Use modern TRL SFTConfig with dynamic fp16/bf16 settings
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=fp16,
        bf16=bf16,
        optim="paged_adamw_8bit", # Memory-efficient optimizer for QLoRA
        max_length=args.max_seq_length,
        report_to="none", # Avoid requiring wandb/tensorboard logins
        dataset_text_field="text" if "text" in dataset.column_names else None,
        packing=False, # Set True if dataset contains short text fields to speed up
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
    )
    
    # 6. Run Training
    logger.info("Starting training loop...")
    trainer.train()
    logger.info("Training complete!")
    
    # 7. Save Final Adapter Weights
    logger.info(f"Saving fine-tuned adapter weights to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Weights saved successfully!")
    
    # 8. Upload to Google Cloud Storage (if specified)
    if args.gcs_bucket:
        upload_directory_to_gcs(args.output_dir, args.gcs_bucket, args.gcs_prefix)
        
    # 9. Push to Hugging Face Hub (if specified)
    if args.hub_model_id:
        logger.info(f"Pushing adapter weights and tokenizer to Hugging Face Hub: {args.hub_model_id}")
        try:
            trainer.model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
            logger.info("Successfully pushed to Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()
