import os
import argparse
import requests
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gemma 4 Sentiment Classification Precision")
    parser.add_argument(
        "--url", 
        type=str, 
        default=None, 
        help="Base URL of the running Cloud Run service (e.g. https://gemma-4-serve-xxxxxx.a.run.app)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run evaluation locally using PyTorch/Transformers (requires local GPU/MPS/CPU and model download)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-4-E4B-it",
        help="Base model ID (if running locally)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to local or GCS adapter folder (if running locally)"
    )
    return parser.parse_args()

# Core test cases representing standard and difficult edge cases (sarcasm, double negation, contrasting clauses)
TEST_CASES = [
    {
        "text": "The software has a steep learning curve, but it is absolutely brilliant!",
        "expected": "Positive",
        "category": "Mixed/But Contrast"
    },
    {
        "text": "Worst service ever, completely broken.",
        "expected": "Negative",
        "category": "Simple Negative"
    },
    {
        "text": "This movie was not without its charm.",
        "expected": "Positive",
        "category": "Double Negation"
    },
    {
        "text": "Oh great, another update that breaks the entire app. Exactly what I needed.",
        "expected": "Negative",
        "category": "Sarcasm/Irony"
    },
    {
        "text": "The delivery was extremely fast.",
        "expected": "Positive",
        "category": "Simple Positive"
    },
    {
        "text": "The book is bound in red leather and has 340 pages.",
        "expected": "Neutral",
        "category": "Domain-Specific Description"
    },
    {
        "text": "I can't say it wasn't a mistake to buy this gadget.",
        "expected": "Negative",
        "category": "Double Negation Negative"
    },
    {
        "text": "It promised to revolutionize my workflow, but it only added frustration.",
        "expected": "Negative",
        "category": "Mixed/But Contrast"
    }
]

def run_remote_evaluation(service_url):
    print("=" * 70)
    print(f"🚀 RUNNING REMOTE EVALUATION AGAINST CLOUD RUN ENDPOINT")
    print(f"🔗 Endpoint: {service_url}")
    print("=" * 70)
    
    health_url = f"{service_url.rstrip('/')}/health"
    try:
        r = requests.get(health_url, timeout=10)
        if r.status_code == 200:
            print(f"✅ Service Health: ONLINE. Details: {r.json()}\n")
        else:
            print(f"⚠️ Service Health returned status {r.status_code}: {r.text}\n")
    except Exception as e:
        print(f"❌ Error connecting to health endpoint: {e}")
        print("Continuing with evaluation run anyway...\n")
        
    chat_url = f"{service_url.rstrip('/')}/v1/chat/completions"
    
    correct = 0
    total = len(TEST_CASES)
    
    for idx, tc in enumerate(TEST_CASES, 1):
        prompt = f"Classify the sentiment: '{tc['text']}'"
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_new_tokens": 10,
            "temperature": 0.0, # Greedy decoding for deterministic precision
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0
        }
        
        try:
            r = requests.post(chat_url, json=payload, headers={"Content-Type": "application/json"})
            if r.status_code == 200:
                result = r.json()
                model_output = result["choices"][0]["message"]["content"].strip()
                
                # Check accuracy
                is_correct = model_output.lower() == tc["expected"].lower()
                status_char = "✅" if is_correct else "❌"
                if is_correct:
                    correct += 1
                    
                print(f"[{idx}/{total}] Category: {tc['category']}")
                print(f"   Input:    \"{tc['text']}\"")
                print(f"   Expected: {tc['expected']}")
                print(f"   Model:    {model_output} {status_char}")
                print("-" * 50)
            else:
                print(f"❌ [{idx}/{total}] Request failed with status {r.status_code}: {r.text}")
                print("-" * 50)
        except Exception as e:
            print(f"❌ [{idx}/{total}] Connection failed: {e}")
            print("-" * 50)
            
    print(f"\n📊 EVALUATION COMPLETE: {correct}/{total} Correct ({correct/total*100:.1f}% Accuracy)")
    if correct == total:
        print("🌟 PERFECT MODEL PRECISION ALIGNMENT ACHIEVED!")
    elif correct >= total * 0.8:
        print("✨ EXCELLENT MODEL PRECISION ALIGNMENT!")
    else:
        print("⚠️ Model sentiment classification precision requires further training or adapter alignment.")
    print("=" * 70)

def run_local_evaluation(model_id, adapter_path):
    print("=" * 70)
    print(f"🚀 RUNNING LOCAL PYTORCH EVALUATION")
    print(f"📦 Model:   {model_id}")
    if adapter_path:
        print(f"🧩 Adapter: {adapter_path}")
    print("=" * 70)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("⏳ Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Unwrap Gemma4ClippableLinear modules
    for name, module in list(base_model.named_modules()):
        if module.__class__.__name__ == "Gemma4ClippableLinear":
            parts = name.split(".")
            parent = base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.linear)
            
    if adapter_path:
        print(f"⏳ Applying adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
        
    model.eval()
    model.config.use_cache = False # Disable cache for dimension safety
    
    if tokenizer.chat_template is None:
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
        
    print("\nStarting local evaluation pass...")
    correct = 0
    total = len(TEST_CASES)
    
    for idx, tc in enumerate(TEST_CASES, 1):
        prompt = f"Classify the sentiment: '{tc['text']}'"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=None,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        model_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean special token bleeding
        if "<" in model_output:
            model_output = model_output.split("<")[0]
        model_output = model_output.strip()
        
        is_correct = model_output.lower() == tc["expected"].lower()
        status_char = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
            
        print(f"[{idx}/{total}] Category: {tc['category']}")
        print(f"   Input:    \"{tc['text']}\"")
        print(f"   Expected: {tc['expected']}")
        print(f"   Model:    {model_output} {status_char}")
        print("-" * 50)
        
    print(f"\n📊 LOCAL EVALUATION COMPLETE: {correct}/{total} Correct ({correct/total*100:.1f}% Accuracy)")
    print("=" * 70)

def main():
    args = parse_args()
    
    if args.local:
        run_local_evaluation(args.model_id, args.adapter_path)
    elif args.url:
        run_remote_evaluation(args.url)
    else:
        print("❌ Error: You must specify either --url <cloud_run_service_url> to run a remote API evaluation,")
        print("   or --local (along with optional --adapter_path) to run local PyTorch validation.")
        print("\nExamples:")
        print("   python evaluation_check.py --url https://gemma-4-serve-xxxxxx.a.run.app")
        print("   python evaluation_check.py --local --adapter_path ./results")
        sys.exit(1)

if __name__ == "__main__":
    main()
