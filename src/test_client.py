import sys
import argparse
import requests
import json
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Test client for Gemma 4 Cloud Run Service")
    parser.add_argument(
        "--url", 
        type=str, 
        required=True, 
        help="Base URL of the running server (e.g. http://localhost:8080 or https://gemma-4-serve-xxxxxx.a.run.app)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["generate", "chat"], 
        default="chat", 
        help="Endpoint mode: 'generate' (/generate) or 'chat' (/v1/chat/completions)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Explain the difference between a base model and an instruction-tuned model in machine learning.", 
        help="Prompt or message to send to the model"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    
    return parser.parse_args()

def test_generate(base_url, prompt, temp, max_tokens):
    url = f"{base_url.rstrip('/')}/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    print(f"\n[Sending Request] POST {url}")
    print(f"[Payload] {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*50)
            print(" MODEL RESPONSE ")
            print("="*50)
            print(data["output"])
            print("="*50)
            print(f"Stats:")
            print(f"  - Time taken: {elapsed:.2f} seconds")
            print(f"  - Prompt tokens: {data['prompt_tokens']}")
            print(f"  - Generated tokens: {data['generated_tokens']}")
            print(f"  - Speed: {data['generated_tokens'] / elapsed:.2f} tokens/sec")
            print("="*50 + "\n")
        else:
            print(f"\n[Error] Status Code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n[Exception] Failed to connect: {e}")

def test_chat(base_url, prompt, temp, max_tokens):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful and extremely intelligent AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_new_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    print(f"\n[Sending Request] POST {url} (OpenAI-Compatible)")
    print(f"[Payload] {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            choice = data["choices"][0]
            print("\n" + "="*50)
            print(" CHAT COMPLETION RESPONSE ")
            print("="*50)
            print(choice["message"]["content"])
            print("="*50)
            print(f"Stats:")
            print(f"  - Time taken: {elapsed:.2f} seconds")
            print(f"  - Prompt tokens: {data['usage']['prompt_tokens']}")
            print(f"  - Generated tokens: {data['usage']['completion_tokens']}")
            print(f"  - Total tokens: {data['usage']['total_tokens']}")
            print(f"  - Speed: {data['usage']['completion_tokens'] / elapsed:.2f} tokens/sec")
            print("="*50 + "\n")
        else:
            print(f"\n[Error] Status Code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n[Exception] Failed to connect: {e}")

def main():
    args = parse_args()
    
    # Check if server is running
    health_url = f"{args.url.rstrip('/')}/health"
    print(f"Checking server health: {health_url} ...")
    try:
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            print(f"Health Check Success: {r.json()}")
        else:
            print(f"Health Check Warning (Status {r.status_code}): {r.text}")
    except Exception as e:
        print(f"Health Check Failed (Continuing anyway): {e}")
        
    if args.mode == "generate":
        test_generate(args.url, args.prompt, args.temperature, args.max_tokens)
    else:
        test_chat(args.url, args.prompt, args.temperature, args.max_tokens)

if __name__ == "__main__":
    main()
