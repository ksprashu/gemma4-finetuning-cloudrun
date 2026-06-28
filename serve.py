import os
import shutil
import tempfile
import urllib.parse
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI Setup
app = FastAPI(
    title="Gemma 4 Inference Service on Cloud Run",
    description="Serve Gemma 4 base, IT, or fine-tuned models with FastAPI",
    version="1.0.0"
)

# Enable CORS for easy cross-origin testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None

# Request/Response schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The raw prompt text to feed the model")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling top_p")
    top_k: int = Field(50, ge=1, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, description="Repetition penalty")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker (e.g. system, user, assistant)")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages in conversational style")
    max_new_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1)
    repetition_penalty: float = Field(1.1, ge=1.0)

class GenerateResponse(BaseModel):
    output: str = Field(..., description="Generated text response")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    generated_tokens: int = Field(..., description="Number of tokens generated")

def download_gcs_directory(gcs_url: str, local_dir: str):
    """Downloads a GCS folder/prefix locally."""
    try:
        from google.cloud import storage
        parsed = urllib.parse.urlparse(gcs_url)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        
        logger.info(f"Downloading adapters from GCS bucket '{bucket_name}' with prefix '{prefix}' to '{local_dir}'...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            raise ValueError(f"No blobs found at GCS path: {gcs_url}")
            
        os.makedirs(local_dir, exist_ok=True)
        for blob in blobs:
            # Skip folders represented as empty blobs
            if blob.name.endswith("/"):
                continue
                
            # Maintain directory structure
            rel_path = os.path.relpath(blob.name, prefix)
            dest_file = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            
            blob.download_to_filename(dest_file)
            logger.info(f"Downloaded: {blob.name} -> {dest_file}")
            
        logger.info("Successfully downloaded all adapter files from GCS.")
    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        raise RuntimeError(f"GCS download failed: {e}")

def resolve_hf_token():
    """Resolves Hugging Face Token from environment or Google Secret Manager."""
    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
        return os.environ["HF_TOKEN"]
    
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

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    
    # Retrieve configuration environment variables
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")
    lora_path = os.environ.get("LORA_ADAPTER_PATH", None)
    
    logger.info(f"Initializing Gemma 4 service. Base Model: {model_id}")
    if lora_path:
        logger.info(f"LoRA Adapter Path: {lora_path}")
        
    # Check for Hugging Face token
    token = resolve_hf_token()
    if not token:
        logger.warning("HF_TOKEN is not defined in environment or Secret Manager. Model download might fail for gated repositories.")
        
    # Configure 4-bit Quantization (essential for deploying on a single L4 GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Check GPU availability
    device_map = "auto"
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Loading on CPU is highly discouraged and will be extremely slow.")
        device_map = "cpu"
        
    # Load base tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Unwrap Gemma4ClippableLinear modules if present to prevent PEFT/LoRA errors
    unwrapped_count = 0
    for name, module in list(base_model.named_modules()):
        if module.__class__.__name__ == "Gemma4ClippableLinear":
            parts = name.split(".")
            parent = base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.linear)
            unwrapped_count += 1
    if unwrapped_count > 0:
        logger.info(f"Successfully unwrapped {unwrapped_count} Gemma4ClippableLinear modules.")
    
    # Load LoRA adapter if specified
    if lora_path:
        local_adapter_dir = lora_path
        
        # If GCS path, download first to a temporary directory
        if lora_path.startswith("gs://"):
            local_adapter_dir = os.path.join(tempfile.gettempdir(), "gemma_adapter")
            # If directory exists, clear it to avoid stale adapter files
            if os.path.exists(local_adapter_dir):
                shutil.rmtree(local_adapter_dir)
            download_gcs_directory(lora_path, local_adapter_dir)
            
        logger.info(f"Applying LoRA adapter from local path: {local_adapter_dir}")
        model = PeftModel.from_pretrained(base_model, local_adapter_dir)
        
        # Override the tokenizer if adapter path has tokenizer files
        if os.path.exists(os.path.join(local_adapter_dir, "tokenizer_config.json")):
            logger.info("Overriding tokenizer with the one saved in adapter folder")
            tokenizer = AutoTokenizer.from_pretrained(local_adapter_dir)
    else:
        model = base_model
        
    # Disable cache to prevent PyTorch dimension mismatch bugs during inference
    model.config.use_cache = False
        
    # Set standard Gemma chat template if not present (essential for handling conversational queries)
    if tokenizer.chat_template is None:
        logger.info("Setting standard Gemma chat template on serving tokenizer.")
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
        
    model.eval()
    logger.info("Gemma 4 Service is ready to receive requests!")

@app.get("/health")
def health_check():
    """Health check endpoint for Cloud Run."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generates text from a raw text prompt."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
        
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None) # Remove token_type_ids if present to avoid generation dimension mismatch
        prompt_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                top_p=request.top_p if request.temperature > 0 else None,
                top_k=request.top_k if request.temperature > 0 else None,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any special token bleeding or trailing markers (e.g. <end_of_turn>\n<)
        if "<" in generated_text:
            generated_text = generated_text.split("<")[0]
        generated_text = generated_text.strip()
        
        return GenerateResponse(
            output=generated_text,
            prompt_tokens=prompt_length,
            generated_tokens=len(generated_ids)
        )
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible Chat Completion endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
        
    try:
        # Convert request to chat list format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply Hugging Face Chat Template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None) # Remove token_type_ids if present to avoid generation dimension mismatch
        prompt_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                top_p=request.top_p if request.temperature > 0 else None,
                top_k=request.top_k if request.temperature > 0 else None,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any special token bleeding or trailing markers (e.g. <end_of_turn>\n<)
        if "<" in generated_text:
            generated_text = generated_text.split("<")[0]
        generated_text = generated_text.strip()
        
        # Formulate OpenAI compatible response
        return {
            "id": "chatcmpl-gemma4",
            "object": "chat.completion",
            "model": os.environ.get("MODEL_ID", "gemma-4"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_length,
                "completion_tokens": len(generated_ids),
                "total_tokens": prompt_length + len(generated_ids)
            }
        }
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable (required by Cloud Run)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("serve:app", host="0.0.0.0", port=port)
