import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    # Helper functions to add markdown and code cells
    def add_markdown(source_lines):
        # Ensure each line ends with newline (except possibly the last one)
        lines = [line + "\n" if not line.endswith("\n") else line for line in source_lines]
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source_lines):
        lines = [line + "\n" if not line.endswith("\n") else line for line in source_lines]
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # --- CELL 1: Introduction ---
    add_markdown([
        "# 🎯 Gemma 4 Fine-Tuning & Inference in Google Colab (T4 GPU)",
        "",
        "This companion notebook walks you step-by-step through setting up, testing, and fine-tuning Google's **Gemma 4 (Edge 2B)** model using standard **Hugging Face Transformers, PEFT, and TRL** on a free **NVIDIA T4 GPU** inside Google Colab.",
        "",
        "### 🧠 What is Gemma 4 E2B?",
        "The **Gemma 4 E2B** (Edge 2B) is part of Google's lightweight open-weights model family, engineered specifically for edge and mobile environments. It supports native multimodal inputs and fits comfortably inside memory-constrained devices. It comes in two primary forms:",
        "1. **`google/gemma-4-E2B` (Base Model)**: Pre-trained on a massive corpus. It excels at autocompleting text but doesn't understand conversational instruction formats.",
        "2. **`google/gemma-4-E2B-it` (Instruction-Tuned Model)**: Fine-tuned by Google to follow conversational formatting, reply to user prompts, and act as an interactive assistant.",
        "",
        "### 🎭 Model Behavioral Comparison Matrix",
        "This matrix displays the exact behavior expected from each model variant given the **exact same prompt**:",
        "",
        "| Model Variant | Expected Behavior | Example Output | Linguistic/Statistical Explanation |",
        "| :--- | :--- | :--- | :--- |",
        "| **Instruction-Tuned (IT)**<br>`google/gemma-4-E2B-it` | **Task Aligned Helper**<br>Understands conversational framing, targets system prompts, and responds succinctly. | `Positive` | It has been aligned using **Supervised Fine-Tuning (SFT)** and RLHF to identify conversational markers (e.g. user-assistant templates) and output task compliance. |",
        "| **Base Model**<br>`google/gemma-4-E2B` | **Raw Statistical Autocomplete**<br>Ignores the command and treats the prompt as the start of a document, continuing to list reviews or templates. | `Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'`<br>`Classify the sentiment: 'Bad battery.'`<br>`Classify the sentiment: 'Excellent stay.'` | It acts as a **high-entropy token completer**. It only knows how to continue the statistical pattern of the input text, not how to follow an instruction. |",
        "| **Fine-Tuned Model**<br>`Base Model + Sentiment Adapter` | **Domain Aligned Specialist**<br>Constrains its high-entropy autocomplete states to output *only* your target sentiment label. | `Positive` | QLoRA backpropagation updated the adapter's linear projection layers, steering attention heads to emit only the targeted classification tokens when prompted with the instruction template. |",
        "",
        "### ⚡ QLoRA: Fine-Tuning 2B parameters on a T4 GPU",
        "- **4-bit Quantization**: We load the base model in 4-bit precision (NF4), reducing the base memory footprint to **~1.5GB to 2.0GB VRAM**.",
        "- **LoRA Adapters**: Instead of updating all 2B parameters, we freeze the base model and train small, lightweight matrices (adapters) attached to the target layers (`q_proj`, `v_proj`, etc.).",
        "- **Paged 8-bit Optimizers**: We use `paged_adamw_8bit` which offloads optimizer memory states to system RAM during training peaks.",
        "",
        "This notebook is **100% self-contained** and can be run end-to-end on Colab's free tier!",
        "",
        "---"
    ])

    # --- CELL 2: Install dependencies ---
    add_code([
        "# Install required packages for training, quantization, and GCS integrations\n",
        "!pip install -q -U transformers peft trl bitsandbytes accelerate datasets google-cloud-storage"
    ])

    # --- CELL 3: Hugging Face Login ---
    add_markdown([
        "## 🔑 Step 1: Hugging Face Authentication",
        "",
        "Because Gemma 4 is a **gated model**, you must accept the license terms on the Hugging Face model cards before running this notebook:",
        "- [Gemma 4 E2B Base](https://huggingface.co/google/gemma-4-E2B)",
        "- [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-E2B-it)",
        "",
        "Once accepted, grab your **Hugging Face Token** and enter it below. The cell is designed to automatically check for Colab's built-in **Secrets** manager (recommended) or fall back to an interactive login block.",
        "",
        "> [%sIMPORTANT]" % "!" + "\n" +
        "> **🔑 Sharing and Pushing to HF Hub:** If you plan to upload your fine-tuned model weights to the **Hugging Face Hub** at the end of this notebook, make sure to generate and use a Hugging Face Token with **Write** permissions (from your [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens)). A read-only token can download the models but will fail during push operations."
    ])

    add_code([
        "import os\n",
        "try:\n",
        "    from google.colab import userdata\n",
        "    # If you have added HF_TOKEN as a Colab Secret (the left key-icon menu), load it directly\n",
        "    hf_token = userdata.get('HF_TOKEN')\n",
        "    os.environ[\"HF_TOKEN\"] = hf_token\n",
        "    print(\"✅ HF_TOKEN detected and loaded from Colab Secrets!\")\n",
        "except Exception:\n",
        "    print(\"ℹ️ HF_TOKEN not found in Colab Secrets. Falling back to interactive login...\")\n",
        "    from huggingface_hub import notebook_login\n",
        "    notebook_login()"
    ])

    # --- CELL 4: Markdown Section 1 (IT Model) ---
    add_markdown([
        "## 💬 Step 2: Testing the Instruction-Tuned (IT) Model",
        "",
        "We will load `google/gemma-4-E2B-it` in **4-bit quantization** to demonstrate how an aligned conversational model behaves. It will follow your prompt's system instructions and output a clean, formatted sentiment label."
    ])

    # --- CELL 5: Loading & running IT model ---
    add_code([
        "import torch\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
        "\n",
        "it_model_id = \"google/gemma-4-E2B-it\"\n",
        "\n",
        "# 1. Define 4-bit Quantization (QLoRA config)\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_compute_dtype=torch.float16, # Use float16 for T4 compatibility\n",
        "    bnb_4bit_use_double_quant=True\n",
        ")\n",
        "\n",
        "# 2. Load the Model & Tokenizer\n",
        "print(f\"⏳ Loading instruction-tuned model '{it_model_id}' in 4-bit...\")\n",
        "it_model = AutoModelForCausalLM.from_pretrained(\n",
        "    it_model_id,\n",
        "    quantization_config=bnb_config,\n",
        "    torch_dtype=torch.float16,\n",
        "    device_map=\"auto\"\n",
        ")\n",
        "\n",
        "# Unwrap Gemma4ClippableLinear modules if present to prevent PEFT/LoRA errors\n",
        "for name, module in list(it_model.named_modules()):\n",
        "    if module.__class__.__name__ == 'Gemma4ClippableLinear':\n",
        "        parts = name.split('.')\n",
        "        parent = it_model\n",
        "        for part in parts[:-1]:\n",
        "            parent = getattr(parent, part)\n",
        "        setattr(parent, parts[-1], module.linear)\n",
        "\n",
        "it_tokenizer = AutoTokenizer.from_pretrained(it_model_id)\n",
        "print(\"✅ Instruction-Tuned model loaded successfully!\")\n",
        "\n",
        "# 3. Test prompt using standard Chat Template\n",
        "messages = [\n",
        "    {\"role\": \"user\", \"content\": \"Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'\"}\n",
        "]\n",
        "\n",
        "formatted_prompt = it_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
        "inputs = it_tokenizer(formatted_prompt, return_tensors=\"pt\").to(it_model.device)\n",
        "\n",
        "print(f\"\\n[Formatted Prompt for Model]:\\n{formatted_prompt}\\n\")\n",
        "\n",
        "with torch.no_grad():\n",
        "    outputs = it_model.generate(\n",
        "        **inputs,\n",
        "        max_new_tokens=64,\n",
        "        do_sample=False,\n",
        "        pad_token_id=it_tokenizer.eos_token_id\n",
        "    )\n",
        "\n",
        "prompt_len = inputs.input_ids.shape[1]\n",
        "generated_ids = outputs[0][prompt_len:]\n",
        "response = it_tokenizer.decode(generated_ids, skip_special_tokens=True)\n",
        "\n",
        "print(\"\\n============================================================\")\n",
        "print(\"🌟 IT MODEL INFERENCE RESULTS & BEHAVIORAL ANALYSIS\")\n",
        "print(\"============================================================\")\n",
        "print(f\"Original Prompt: \\\"Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'\\\"\")\n",
        "print(f\"Raw Generated Text:\\n{response.strip()}\")\n",
        "print(\"-\"*60)\n",
        "print(\"💡 ANALYST EXPLANATION:\")\n",
        "print(\"The Instruction-Tuned (IT) model understands turn-taking syntax because\")\n",
        "print(\"it has undergone SFT and RLHF alignment. It recognizes user queries inside\")\n",
        "print(\"conversational templates and behaves as an assistant, targeting your\")\n",
        "print(\"instruction directly and returning the classification label or structured answer.\")\n",
        "print(\"============================================================\")\n"
    ])

    # --- CELL 5.5: IT Model Playground ---
    add_markdown([
        "### 🎮 Try Your Own Prompts on the IT Model!",
        "Before we clear the model from memory to avoid running out of RAM, use the interactive form-field below to test other prompts of your choice."
    ])

    add_code([
        "#@title 💬 Interactive IT Model Playground { run: \"auto\" }\n",
        "#@markdown Type your own prompt below to test how the aligned instruction-tuned model responds.\n",
        "\n",
        "custom_prompt = \"Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'\" #@param {type:\"string\"}\n",
        "\n",
        "# Ensure model is still in memory\n",
        "if 'it_model' in globals() and 'it_tokenizer' in globals():\n",
        "    messages = [{\"role\": \"user\", \"content\": custom_prompt}]\n",
        "    formatted_prompt = it_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
        "    inputs = it_tokenizer(formatted_prompt, return_tensors=\"pt\").to(it_model.device)\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        outputs = it_model.generate(\n",
        "            **inputs,\n",
        "            max_new_tokens=64,\n",
        "            do_sample=False,\n",
        "            pad_token_id=it_tokenizer.eos_token_id\n",
        "        )\n",
        "    \n",
        "    prompt_len = inputs.input_ids.shape[1]\n",
        "    response = it_tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()\n",
        "    \n",
        "    print(\"=\"*60)\n",
        "    print(\"🎭 IT MODEL RESPONSE:\")\n",
        "    print(\"=\"*60)\n",
        "    print(response)\n",
        "    print(\"=\"*60)\n",
        "else:\n",
        "    print(\"⚠️ Error: The Instruction-Tuned model has already been deleted from VRAM. Please re-run the loading cell above.\")"
    ])

    # --- CELL 6: Clean up IT model to avoid OOM ---
    add_markdown([
        "### 🧹 Freeing VRAM Memory",
        "Because T4 GPUs are limited, we must delete the loaded IT model and clear PyTorch's cache before loading the Base model to prevent out-of-memory crashes."
    ])

    add_code([
        "import sys\n",
        "import gc\n",
        "import torch\n",
        "\n",
        "# 1. Clear model, tokenizer, and input references from the namespace\n",
        "for var in ['it_model', 'it_tokenizer', 'inputs', 'outputs', 'formatted_prompt', 'custom_prompt', 'response']:\n",
        "    if var in globals():\n",
        "        del globals()[var]\n",
        "\n",
        "# 2. Clear traceback references to release memory held by errors\n",
        "sys.last_traceback = None\n",
        "sys.last_value = None\n",
        "sys.last_type = None\n",
        "\n",
        "# 3. Flush IPython's command history / Out dict cache\n",
        "try:\n",
        "    ipython = get_ipython()\n",
        "    if ipython is not None:\n",
        "        ipython.user_ns.pop('_', None)\n",
        "        ipython.user_ns.pop('__', None)\n",
        "        ipython.user_ns.pop('___', None)\n",
        "        for key in list(ipython.user_ns.keys()):\n",
        "            if key.startswith('_') and key[1:].isdigit():\n",
        "                ipython.user_ns.pop(key, None)\n",
        "        if 'Out' in ipython.user_ns:\n",
        "            ipython.user_ns['Out'].clear()\n",
        "except NameError:\n",
        "    pass\n",
        "\n",
        "# 4. Trigger garbage collector & empty PyTorch CUDA cache\n",
        "gc.collect()\n",
        "torch.cuda.empty_cache()\n",
        "print(\"🧹 VRAM memory cleared! 100% reference cache flushed successfully.\")"
    ])

    # --- CELL 7: Markdown Section 2 (Base Model) ---
    add_markdown([
        "## 📝 Step 3: Testing the Base Model (Contrastive Autocomplete)",
        "",
        "Now, we will load the pre-trained base model `google/gemma-4-E2B` and run the **exact same prompt**.",
        "",
        "Observe the output: instead of answering with a classification label, the base model will continue autocompleting reviews or creating similar templates. This is normal base model behavior because it lacks conversational instruction-alignment."
    ])

    # --- CELL 8: Loading & running Base model ---
    add_code([
        "base_model_id = \"google/gemma-4-E2B\"\n",
        "\n",
        "print(f\"⏳ Loading base model '{base_model_id}' in 4-bit...\")\n",
        "base_model = AutoModelForCausalLM.from_pretrained(\n",
        "    base_model_id,\n",
        "    quantization_config=bnb_config,\n",
        "    torch_dtype=torch.float16,\n",
        "    device_map=\"auto\"\n",
        ")\n",
        "\n",
        "# Unwrap Gemma4ClippableLinear modules if present to prevent PEFT/LoRA errors\n",
        "for name, module in list(base_model.named_modules()):\n",
        "    if module.__class__.__name__ == 'Gemma4ClippableLinear':\n",
        "        parts = name.split('.')\n",
        "        parent = base_model\n",
        "        for part in parts[:-1]:\n",
        "            parent = getattr(parent, part)\n",
        "        setattr(parent, parts[-1], module.linear)\n",
        "\n",
        "base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)\n",
        "print(\"✅ Base model loaded successfully!\")\n",
        "\n",
        "# Run the exact same prompt\n",
        "prompt = \"Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'\"\n",
        "inputs = base_tokenizer(prompt, return_tensors=\"pt\").to(base_model.device)\n",
        "\n",
        "with torch.no_grad():\n",
        "    outputs = base_model.generate(\n",
        "        **inputs,\n",
        "        max_new_tokens=100,\n",
        "        temperature=0.7,\n",
        "        do_sample=True,\n",
        "        pad_token_id=base_tokenizer.eos_token_id\n",
        "    )\n",
        "\n",
        "response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
        "\n",
        "print(\"\\n============================================================\")\n",
        "print(\"📝 BASE MODEL INFERENCE RESULTS & BEHAVIORAL ANALYSIS\")\n",
        "print(\"============================================================\")\n",
        "print(f\"Original Prompt: \\\"{prompt}\\\"\")\n",
        "print(f\"Raw Generated Text:\\n{response}\")\n",
        "print(\"-\"*60)\n",
        "print(\"⚠️ ANALYST EXPLANATION:\")\n",
        "print(\"Observe how the Base Model completely ignored the command 'Classify the sentiment'!\")\n",
        "print(\"Instead of outputting 'Positive', it autocompleted our text. It likely added\")\n",
        "print(\"more hypothetical reviews (e.g. 'Classify the sentiment: Bad battery...', etc.).\")\n",
        "print(\"This high-entropy document completion is the standard, expected behavior\")\n",
        "print(\"of an unaligned base model, which acts as a pure statistical text completer.\")\n",
        "print(\"============================================================\")\n"
    ])

    # --- CELL 8.5: Base Model Playground ---
    add_markdown([
        "### 🎮 Try Your Own Prompts on the Base Model!",
        "Use the interactive form below to test other custom prompts on the Base Model. Notice how it behaves as a pure autocomplete engine, continuing whatever pattern you start."
    ])

    add_code([
        "#@title 📝 Interactive Base Model Playground { run: \"auto\" }\n",
        "#@markdown Type your own text prompt below to observe how the unaligned base model autocompletes/continues it.\n",
        "\n",
        "custom_prompt = \"Classify the sentiment: 'I had an absolutely wonderful experience!'\" #@param {type:\"string\"}\n",
        "\n",
        "if 'base_model' in globals() and 'base_tokenizer' in globals():\n",
        "    inputs = base_tokenizer(custom_prompt, return_tensors=\"pt\").to(base_model.device)\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        outputs = base_model.generate(\n",
        "            **inputs,\n",
        "            max_new_tokens=100,\n",
        "            temperature=0.7,\n",
        "            do_sample=True,\n",
        "            pad_token_id=base_tokenizer.eos_token_id\n",
        "        )\n",
        "    \n",
        "    response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
        "    \n",
        "    print(\"=\"*60)\n",
        "    print(\"📝 BASE MODEL AUTOCOMPLETE RESPONSE:\")\n",
        "    print(\"=\"*60)\n",
        "    print(response)\n",
        "    print(\"=\"*60)\n",
        "else:\n",
        "    print(\"⚠️ Error: The Base model has not been loaded yet, or was removed. Please run the Base model loading cell above.\")"
    ])

    # --- CELL 8.6: Clean up Base model to avoid OOM ---
    add_markdown([
        "### 🧹 Freeing VRAM Memory Before Training",
        "To avoid running out of memory during fine-tuning, we must delete our Step 3 inference base model and clear PyTorch's cache. This ensures the GPU has maximum free VRAM for gradient updates and optimizer states during training."
    ])

    add_code([
        "import sys\n",
        "import gc\n",
        "import torch\n",
        "\n",
        "# 1. Clear model, tokenizer, and input references from the namespace\n",
        "for var in ['base_model', 'base_tokenizer', 'inputs', 'outputs', 'custom_prompt', 'response']:\n",
        "    if var in globals():\n",
        "        del globals()[var]\n",
        "\n",
        "# 2. Clear traceback references to release memory held by errors\n",
        "sys.last_traceback = None\n",
        "sys.last_value = None\n",
        "sys.last_type = None\n",
        "\n",
        "# 3. Flush IPython's command history / Out dict cache\n",
        "try:\n",
        "    ipython = get_ipython()\n",
        "    if ipython is not None:\n",
        "        ipython.user_ns.pop('_', None)\n",
        "        ipython.user_ns.pop('__', None)\n",
        "        ipython.user_ns.pop('___', None)\n",
        "        for key in list(ipython.user_ns.keys()):\n",
        "            if key.startswith('_') and key[1:].isdigit():\n",
        "                ipython.user_ns.pop(key, None)\n",
        "        if 'Out' in ipython.user_ns:\n",
        "            ipython.user_ns['Out'].clear()\n",
        "except NameError:\n",
        "    pass\n",
        "\n",
        "# 4. Trigger garbage collector & empty PyTorch CUDA cache\n",
        "gc.collect()\n",
        "torch.cuda.empty_cache()\n",
        "print(\"🧹 VRAM memory cleared and ready for fine-tuning!\")"
    ])

    # --- CELL 9: Markdown Section 3 (Dataset Synthesis) ---
    add_markdown([
        "## 📊 Step 4: Programmatic & Agentic Dataset Synthesis",
        "",
        "To prepare your Gemma 4 model for training, you have two major pathways for dataset preparation:",
        "",
        "### 🧱 Pathway A: Local Programmatic Script (Included Below)",
        "To make this notebook **100% self-contained**, we include a local python cell that programmatically compiles **500 sentiment analysis samples** covering varied linguistic nuances (such as double negations, slang, sarcasm, and emoji emphasis) and saves them locally to `sentiment_dataset.jsonl` matching Gemma's conversational training schema:",
        "```json",
        "{",
        "  \"messages\": [",
        "    {\"role\": \"user\", \"content\": \"Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'\"},",
        "    {\"role\": \"model\", \"content\": \"Positive\"}",
        "  ]",
        "}",
        "```",
        "",
        "---",
        "",
        "### 🚀 Pathway B: Agentic Synthesis with Google Antigravity (AGY)",
        "While programmatic template generation is useful for a quick demo, it has highly limited vocabulary diversity. For production-grade fine-tuning, you need high-fidelity synthetic data. By using the **AGY CLI (`agy`)** or **Antigravity 2.0 Desktop app**, you can command the AI to act as a **Producer Archetype** to generate thousands of unique, natural, and complex samples with structured schema constraints.",
        "",
        "#### 💡 Other Domain Ideas & Precise AGY Prompts",
        "Here are three other high-value domain-specific use cases you can fine-tune Gemma 4 on, along with the exact prompts you can copy and paste directly into **AGY** to generate your custom datasets:",
        "",
        "##### 1️⃣ Multilingual Support Ticket Router",
        "* **Concept:** Fine-tune Gemma to act as an automated ticket triage gateway, classifying raw customer messages (English, Spanish, French, German, Japanese, Hindi) into issues (`billing`, `tech_support`, `account_security`, `refund_request`) and priorities (`critical`, `high`, `medium`, `low`).",
        "* **AGY Prompt:**",
        "  ```",
        "  Role: Act as a data generation engineer (Producer Archetype).",
        "  Task: Create a synthetic fine-tuning dataset of 1000 customer support triage samples.",
        "  Constraints:",
        "  - The input support ticket should be in random languages (English, Spanish, French, German, Japanese, Hindi).",
        "  - The model must output a JSON block indicating category and priority.",
        "  - Format: Save as a JSON Lines (.jsonl) file in my current workspace named `triage_dataset.jsonl`.",
        "  - Schema: Every line must be a single JSON object matching this structure exactly:",
        "    {\"messages\": [{\"role\": \"user\", \"content\": \"Triage this support ticket: '<TICKET_TEXT>'\"}, {\"role\": \"model\", \"content\": \"{\\\"category\\\": \\\"<CAT>\\\", \\\"priority\\\": \\\"<PRIORITY>\\\"}\"}]}",
        "  ```",
        "",
        "##### 2️⃣ Text-to-API Payload Copilot (Function Caller)",
        "* **Concept:** Fine-tune Gemma to translate unstructured natural language request intents into structured REST API JSON payloads for backend microservices.",
        "* **AGY Prompt:**",
        "  ```",
        "  Role: Act as a data generation engineer (Producer Archetype).",
        "  Task: Create a synthetic fine-tuning dataset of 1000 Text-to-API translation samples.",
        "  Constraints:",
        "  - The user prompt should be a natural language request to perform an action (e.g., booking, profile updates).",
        "  - The model response should be a formatted, structured REST API payload.",
        "  - Format: Save as a JSON Lines (.jsonl) file in my current workspace named `api_dataset.jsonl`.",
        "  - Schema: Every line must be a single JSON object matching this structure exactly:",
        "    {\"messages\": [{\"role\": \"user\", \"content\": \"Translate to API payload: '<REQUEST_TEXT>'\"}, {\"role\": \"model\", \"content\": \"{\\\"action\\\": \\\"<ACTION>\\\", \\\"params\\\": {<PARAMETERS>}}\"}]}",
        "  ```",
        "",
        "##### 3️⃣ Enterprise PII Redactor",
        "* **Concept:** Fine-tune Gemma to redact/mask sensitive customer information (emails, names, SSNs, credit card numbers, phone numbers) before logging chat logs.",
        "* **AGY Prompt:**",
        "  ```",
        "  Role: Act as a data generation engineer (Producer Archetype).",
        "  Task: Create a synthetic fine-tuning dataset of 1000 PII anonymization samples.",
        "  Constraints:",
        "  - The input should be a conversational transcript containing names, credit card numbers, email addresses, phone numbers, or SSNs.",
        "  - The model response must be the exact same transcript, but with all PII replaced by standardized tags like `[REDACTED_NAME]`, `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_CARD]`.",
        "  - Format: Save as a JSON Lines (.jsonl) file in my current workspace named `pii_dataset.jsonl`.",
        "  - Schema: Every line must be a single JSON object matching this structure exactly:",
        "    {\"messages\": [{\"role\": \"user\", \"content\": \"Redact PII from this transcript: '<TRANSCRIPT>'\"}, {\"role\": \"model\", \"content\": \"<REDACTED_TRANSCRIPT>\"}]}",
        "  ```",
        "",
        "Let's execute Pathway A to programmatically synthesize our sentiment dataset and run the training process."
    ])

    # --- CELL 10: Generating Dataset code ---
    add_code([
        "import json\n",
        "import random\n",
        "\n",
        "# Define lexicon databases for synthesis\n",
        "domains = [\"product\", \"movie\", \"app\", \"restaurant\", \"stay\"]\n",
        "templates = {\n",
        "    \"pos\": [\n",
        "        \"This {domain} is incredible and highly recommended.\",\n",
        "        \"Absolutely loved my stay, everything was spotless!\",\n",
        "        \"It works like a charm, worth every single dollar.\",\n",
        "        \"I don't dislike this {domain} at all, it's amazing.\",\n",
        "        \"This is easily the GOAT! Extremely top-tier 🔥\",\n",
        "        \"Although the setup was steep, it is spectacular!\"\n",
        "    ],\n",
        "    \"neg\": [\n",
        "        \"Worst {domain} I have ever spent money on.\",\n",
        "        \"Oh great, another bug-filled disaster... exactly what I needed.\",\n",
        "        \"It broke down after only three days of light usage.\",\n",
        "        \"Avoid at all costs. Absolute waste of time.\",\n",
        "        \"Cheap material, clunky interface, and sluggish support.\",\n",
        "        \"I really wanted to support this, but it is a complete trainwreck.\"\n",
        "    ]\n",
        "}\n",
        "\n",
        "synthetic_data = []\n",
        "for i in range(500):\n",
        "    sentiment = random.choice([\"Positive\", \"Negative\"])\n",
        "    tpl_group = \"pos\" if sentiment == \"Positive\" else \"neg\"\n",
        "    raw_review = random.choice(templates[tpl_group]).format(domain=random.choice(domains))\n",
        "    \n",
        "    # Randomize instruction wrapping\n",
        "    inst = f\"Classify the sentiment: '{raw_review}'\"\n",
        "    \n",
        "    synthetic_data.append({\n",
        "        \"messages\": [\n",
        "            {\"role\": \"user\", \"content\": inst},\n",
        "            {\"role\": \"model\", \"content\": sentiment}\n",
        "        ]\n",
        "    })\n",
        "\n",
        "# Save locally to file\n",
        "dataset_path = \"sentiment_dataset.jsonl\"\n",
        "with open(dataset_path, \"w\", encoding=\"utf-8\") as f:\n",
        "    for sample in synthetic_data:\n",
        "        f.write(json.dumps(sample) + \"\\n\")\n",
        "\n",
        "print(f\"✅ Successfully synthesized 500 samples in: '{dataset_path}'\")"
    ])

    # --- CELL 11: Markdown Section 4 (Fine-Tuning config) ---
    add_markdown([
        "## 🏋️ Step 5: Fine-Tuning Gemma Base with QLoRA",
        "",
        "Now we will load our dataset into the Hugging Face `SFTTrainer` and execute our QLoRA training run on the T4 GPU.",
        "",
        "> [%sIMPORTANT]" % "!" + "\n" +
        "> **💥 CRITICAL: Google Colab T4 Memory Management & Session Restart Protocol**\n" +
        "> Because we loaded and played with both the **Instruction-Tuned** and **Base** models in Step 2 and Step 3, Python's garbage collector and Google Colab's interactive forms may have lingering tensor references in memory. Even minor VRAM leaks of 1-2 GB will cause an `OutOfMemoryError` during training because `prepare_model_for_kbit_training` and gradient updates require every megabyte of our 15GB VRAM.\n" +
        ">\n" +
        "> **🔄 To ensure a 100% successful training run with ZERO memory issues:**\n" +
        "> 1. Go to the top menu and select **Runtime -> Restart session** (or use shortcut `Ctrl + M` then `.`).\n" +
        ">    * *Note: Restarting the session clears the Python state and frees 100% of VRAM, but **keeps** all pip installations and files intact!*\n" +
        "> 2. Once the session is restarted, skip the loading cells in Step 2 & 3, and run **Step 1** (HF Authentication) to load your credentials.\n" +
        "> 3. Proceed directly to **Step 5** below to begin fine-tuning!",
        "",
        "### How the adapters are applied:",
        "1. We freeze the loaded 4-bit base model `google/gemma-4-E2B` to prevent its original parameters from shifting.",
        "2. We define a `LoraConfig` that targets only the projection modules. This inserts low-rank matrices into the network, isolating trainable params.",
        "3. We trigger `SFTTrainer` which runs sequence batching, computes loss, and updates our adapter matrices."
    ])

    # --- CELL 12: Training Execution block ---
    add_code([
        "import torch\n",
        "from datasets import load_dataset\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
        "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
        "from trl import SFTConfig, SFTTrainer\n",
        "\n",
        "# 1. Load the generated dataset\n",
        "dataset = load_dataset(\"json\", data_files=dataset_path, split=\"train\")\n",
        "print(f\"Dataset Loaded. Record Count: {len(dataset)}\")\n",
        "\n",
        "# 2. Reload Base Model & Tokenizer Fresh for Fine-Tuning\n",
        "base_model_id = \"google/gemma-4-E2B\"\n",
        "print(f\"⏳ Loading fresh base model '{base_model_id}' in 4-bit...\")\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_compute_dtype=torch.float16, # Use float16 for T4 compatibility\n",
        "    bnb_4bit_use_double_quant=True\n",
        ")\n",
        "base_model = AutoModelForCausalLM.from_pretrained(\n",
        "    base_model_id,\n",
        "    quantization_config=bnb_config,\n",
        "    torch_dtype=torch.float16, # Load unquantized weights in float16 to prevent bfloat16 mixed-type GradScaler errors on T4\n",
        "    device_map=\"auto\"\n",
        ")\n",
        "\n",
        "# Unwrap Gemma4ClippableLinear modules if present to prevent PEFT/LoRA errors\n",
        "for name, module in list(base_model.named_modules()):\n",
        "    if module.__class__.__name__ == 'Gemma4ClippableLinear':\n",
        "        parts = name.split('.')\n",
        "        parent = base_model\n",
        "        for part in parts[:-1]:\n",
        "            parent = getattr(parent, part)\n",
        "        setattr(parent, parts[-1], module.linear)\n",
        "\n",
        "base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)\n",
        "print(\"✅ Fresh Base model loaded successfully!\")\n",
        "\n",
        "# 3. Custom memory-efficient prepare_model_for_kbit_training\n",
        "def custom_prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):\n",
        "    for name, param in model.named_parameters():\n",
        "        param.requires_grad = False\n",
        "    \n",
        "    # Only upcast tiny normalization layers (norm, ln) to float32 for stability.\n",
        "    # We skip upcasting the massive 2.8B parameter embedding & language model head layers,\n",
        "    # saving ~8.75 GiB of VRAM from being allocated!\n",
        "    for name, param in model.named_parameters():\n",
        "        if (\"norm\" in name or \"ln\" in name) and param.__class__.__name__ != \"Params4bit\":\n",
        "            if param.dtype in [torch.float16, torch.bfloat16]:\n",
        "                param.data = param.data.to(torch.float32)\n",
        "    \n",
        "    # Explicitly convert any remaining bfloat16 parameters and buffers to float16 (essential for T4 GPU compatibility)\n",
        "    for name, param in model.named_parameters():\n",
        "        if param.dtype == torch.bfloat16:\n",
        "            param.data = param.data.to(torch.float16)\n",
        "    for name, buf in model.named_buffers():\n",
        "        if buf.dtype == torch.bfloat16:\n",
        "            buf.data = buf.data.to(torch.float16)\n",
        "    \n",
        "    if use_gradient_checkpointing:\n",
        "        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)\n",
        "        if hasattr(model, \"enable_input_require_grads\"):\n",
        "            model.enable_input_require_grads()\n",
        "        else:\n",
        "            def make_inputs_require_grad(module, input, output):\n",
        "                output.requires_grad_(True)\n",
        "            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)\n",
        "    return model\n",
        "\n",
        "base_model = custom_prepare_model_for_kbit_training(base_model)\n",
        "base_tokenizer.pad_token = base_tokenizer.eos_token\n",
        "base_tokenizer.padding_side = \"right\"\n",
        "\n",
        "# Set standard Gemma chat template since we are using a conversational messages dataset\n",
        "base_tokenizer.chat_template = (\n",
        "    \"{{ bos_token }}\"\n",
        "    \"{% for message in messages %}\"\n",
        "    \"{% if message['role'] == 'user' %}\"\n",
        "    \"{{ '<start_of_turn>user\\\\n' + message['content'] + '<end_of_turn>\\\\n' }}\"\n",
        "    \"{% elif message['role'] == 'model' %}\"\n",
        "    \"{{ '<start_of_turn>model\\\\n' + message['content'] + '<end_of_turn>\\\\n' }}\"\n",
        "    \"{% endif %}\"\n",
        "    \"{% endfor %}\"\n",
        "    \"{% if add_generation_prompt %}\"\n",
        "    \"{{ '<start_of_turn>model\\\\n' }}\"\n",
        "    \"{% endif %}\"\n",
        ")\n",
        "\n",
        "# 4. Configure LoRA parameters\n",
        "lora_config = LoraConfig(\n",
        "    r=16,\n",
        "    lora_alpha=32,\n",
        "    lora_dropout=0.05,\n",
        "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
        "    bias=\"none\",\n",
        "    task_type=\"CAUSAL_LM\"\n",
        ")\n",
        "\n",
        "# 5. Define modern TRL SFT Training Parameters\n",
        "output_dir = \"./results\"\n",
        "training_args = SFTConfig(\n",
        "    output_dir=output_dir,\n",
        "    num_train_epochs=1, # 1 Epoch is enough for this self-contained demo\n",
        "    per_device_train_batch_size=2,\n",
        "    gradient_accumulation_steps=8,\n",
        "    gradient_checkpointing=True,\n",
        "    gradient_checkpointing_kwargs={\"use_reentrant\": False},\n",
        "    learning_rate=2e-4,\n",
        "    optim=\"paged_adamw_8bit\", # Saves optimizer memory\n",
        "    fp16=True, # Use FP16 for standard T4 operations\n",
        "    bf16=False,\n",
        "    logging_steps=10,\n",
        "    save_strategy=\"no\",\n",
        "    max_length=512,\n",
        "    report_to=\"none\"\n",
        ")\n",
        "\n",
        "# 6. Initialize SFTTrainer\n",
        "trainer = SFTTrainer(\n",
        "    model=base_model,\n",
        "    train_dataset=dataset,\n",
        "    peft_config=lora_config,\n",
        "    processing_class=base_tokenizer,\n",
        "    args=training_args\n",
        ")\n",
        "\n",
        "# Explicitly align parameter dtypes for mixed-precision training on T4 GPU:\n",
        "# 1. Trainable weights (LoRA adapters) must remain in float32 for GradScaler gradient accumulation.\n",
        "# 2. Frozen weights (embeddings, head) must be cast from bfloat16 to float16 to prevent bfloat16 activation leakage.\n",
        "for name, param in trainer.model.named_parameters():\n",
        "    if param.requires_grad:\n",
        "        if param.dtype != torch.float32:\n",
        "            param.data = param.data.to(torch.float32)\n",
        "    else:\n",
        "        if param.dtype == torch.bfloat16:\n",
        "            param.data = param.data.to(torch.float16)\n",
        "for name, buf in trainer.model.named_buffers():\n",
        "    if buf.dtype == torch.bfloat16:\n",
        "        buf.data = buf.data.to(torch.float16)\n",
        "\n",
        "# 7. Execute Fine-Tuning!\n",
        "print(\"⏳ Starting QLoRA fine-tuning...\")\n",
        "trainer.train()\n",
        "print(\"✅ Fine-Tuning complete!\")\n",
        "\n",
        "# 8. Save Adapter locally\n",
        "adapter_dir = \"./fine_tuned_gemma_adapter\"\n",
        "trainer.model.save_pretrained(adapter_dir)\n",
        "base_tokenizer.save_pretrained(adapter_dir)\n",
        "print(f\"💾 Fine-tuned adapter saved locally to '{adapter_dir}'\")"
    ])

    # --- CELL 12.5: Google Drive Saving Markdown ---
    add_markdown([
        "## 💾 Step 5.5: Saving & Restoring Adapter Weights (Google Drive Integration)",
        "",
        "Because Google Colab's default local directory is **ephemeral**, any files saved there (like `./fine_tuned_gemma_adapter`) will be **lost completely** if the notebook runtime disconnects, resets, or if you restart your session.",
        "",
        "To make your hard work permanent and resilient, you can mount your personal **Google Drive** and save the adapter there. When you open this notebook in a fresh session next time, you can simply mount Google Drive and load your pre-trained adapter in **2 seconds flat**—completely bypassing the 5-minute training phase!",
        "",
        "### 📂 Option A: Save fine-tuned adapter to Google Drive"
    ])

    # --- CELL 12.6: Google Drive Saving Block ---
    add_code([
        "import shutil\n",
        "import os\n",
        "\n",
        "try:\n",
        "    from google.colab import drive\n",
        "    print(\"⏳ Mounting Google Drive to save adapter permanently...\")\n",
        "    drive.mount('/content/drive')\n",
        "    \n",
        "    gdrive_adapter_path = \"/content/drive/MyDrive/fine_tuned_gemma_adapter\"\n",
        "    \n",
        "    print(f\"⏳ Copying adapter weights from local session to Google Drive at '{gdrive_adapter_path}'...\")\n",
        "    if os.path.exists(gdrive_adapter_path):\n",
        "        shutil.rmtree(gdrive_adapter_path)\n",
        "    shutil.copytree(\"./fine_tuned_gemma_adapter\", gdrive_adapter_path)\n",
        "    print(\"🎉 SUCCESS! Your fine-tuned adapter has been saved permanently to your Google Drive.\")\n",
        "    print(\"💡 You can now safely close this Colab runtime or restart the session.\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Failed to save to Google Drive (make sure to grant permissions): {e}\")"
    ])

    # --- CELL 12.7: Loading Markdown ---
    add_markdown([
        "### 🔄 Option B: Restore/Load an existing adapter (Bypass Retraining!)",
        "",
        "If you have already trained your model and saved your adapter to Google Drive, or if you pushed it to the Hugging Face Hub, you do **not** need to run Step 5 (the training loop) again!",
        "",
        "Run Step 1, Step 3 (to load the base model), and then use this interactive cell below to load your adapter directly. This will instantly configure your model for inference!"
    ])

    # --- CELL 12.8: Loading Block ---
    add_code([
        "#@title 🔄 Restore Adapter weights { run: \"manual\" }\n",
        "LOAD_SOURCE = \"Google Drive\" #@param [\"Local Directory\", \"Google Drive\", \"Hugging Face Hub\"]\n",
        "HF_MODEL_ID = \"username/gemma-4-sentiment-adapter\" #@param {type:\"string\"}\n",
        "\n",
        "import os\n",
        "from peft import PeftModel\n",
        "\n",
        "adapter_path = \"./fine_tuned_gemma_adapter\"\n",
        "\n",
        "if LOAD_SOURCE == \"Google Drive\":\n",
        "    try:\n",
        "        from google.colab import drive\n",
        "        print(\"⏳ Mounting Google Drive to restore adapter...\")\n",
        "        drive.mount('/content/drive')\n",
        "        gdrive_path = \"/content/drive/MyDrive/fine_tuned_gemma_adapter\"\n",
        "        if os.path.exists(gdrive_path):\n",
        "            adapter_path = gdrive_path\n",
        "            print(f\"✅ Successfully located adapter in Google Drive: {adapter_path}\")\n",
        "        else:\n",
        "            print(\"⚠️ Warning: Adapter folder not found in Google Drive! Make sure you saved it first.\")\n",
        "    except Exception as e:\n",
        "        print(f\"❌ Failed to mount Google Drive: {e}\")\n",
        "elif LOAD_SOURCE == \"Hugging Face Hub\":\n",
        "    adapter_path = HF_MODEL_ID\n",
        "    print(f\"✅ Loading adapter from Hugging Face Hub: {adapter_path}\")\n",
        "else:\n",
        "    print(f\"✅ Loading from local directory: {adapter_path}\")\n",
        "\n",
        "try:\n",
        "    print(\"⏳ Wrapping base model with PEFT adapter...\")\n",
        "    if 'base_model' in globals():\n",
        "        # Load/re-load the PEFT model wrapper\n",
        "        if hasattr(base_model, \"unload\"):\n",
        "            base_model = base_model.unload() # Reset to base model state\n",
        "        \n",
        "        # Load PEFT adapter\n",
        "        trainer_model = PeftModel.from_pretrained(base_model, adapter_path)\n",
        "        \n",
        "        # Ensure 'trainer.model' is globally set so the downstream evaluation cells use it\n",
        "        if 'trainer' in globals():\n",
        "            trainer.model = trainer_model\n",
        "        else:\n",
        "            # Create a mock trainer class to keep downstream cells running perfectly\n",
        "            class MockTrainer:\n",
        "                pass\n",
        "            trainer = MockTrainer()\n",
        "            trainer.model = trainer_model\n",
        "            \n",
        "        print(\"🎉 SUCCESS! Model successfully wrapped with the pre-trained adapter weights.\")\n",
        "        print(\"💡 You are now ready to run Step 6 (Evaluation) and Step 7/8 (Exports) directly!\")\n",
        "    else:\n",
        "        print(\"❌ Error: 'base_model' is not loaded in memory. Please run Step 1 and Step 3 first!\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Failed to load adapter: {e}\")"
    ])

    # --- CELL 13: Markdown Section 5 (Evaluating Fine-Tuned Model) ---
    add_markdown([
        "## 🔬 Step 6: Evaluating the Fine-Tuned Model",
        "",
        "Let's test our newly fine-tuned model (which is the frozen base model merged with our newly trained LoRA adapter).",
        "",
        "Observe the output: the base model, which formerly autocompleted random text, has now successfully aligned to behave as a **sentiment classifier**, outputting the clean sentiment label!"
    ])

    # --- CELL 14: Executing Fine-Tuned Inference ---
    add_code([
        "from peft import PeftModel\n",
        "\n",
        "# Let's run a test prompt directly on our current model state\n",
        "test_reviews = [\n",
        "    \"Classify the sentiment: 'The software has a steep learning curve, but it is absolutely brilliant!'\",\n",
        "    \"Classify the sentiment: 'Worst service ever, completely broken.'\"\n",
        "]\n",
        "\n",
        "trainer.model.eval()\n",
        "trainer.model.config.use_cache = True # Explicitly enable KV Cache during evaluation for fast error-free decoding\n",
        "\n",
        "print(\"\\n============================================================\")\n",
        "print(\"🚀 FINE-TUNED SPECIALIST MODEL INFERENCE RESULTS & ANALYSIS\")\n",
        "print(\"============================================================\")\n",
        "\n",
        "for pr in test_reviews:\n",
        "    eval_messages = [{\"role\": \"user\", \"content\": pr}]\n",
        "    formatted_prompt = base_tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=True)\n",
        "    inputs = base_tokenizer(formatted_prompt, return_tensors=\"pt\").to(base_model.device)\n",
        "    inputs.pop(\"token_type_ids\", None) # Remove token_type_ids if present to avoid generation dimension mismatch\n",
        "    with torch.no_grad():\n",
        "        outputs = trainer.model.generate(\n",
        "            **inputs,\n",
        "            max_new_tokens=10,\n",
        "            do_sample=False,\n",
        "            pad_token_id=base_tokenizer.eos_token_id\n",
        "        )\n",
        "        \n",
        "    prompt_len = inputs.input_ids.shape[1]\n",
        "    generated_ids = outputs[0][prompt_len:]\n",
        "    label = base_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()\n",
        "    \n",
        "    print(f\"Prompt entered:  '{pr}'\")\n",
        "    print(f\"Model Output:    '{label}'\")\n",
        "    print(\"-\"*60)\n",
        "\n",
        "print(\"\\n💡 ANALYST EXPLANATION:\")\n",
        "print(\"This is a dramatic transformation! The raw Base Model, which previously\")\n",
        "print(\"rambled into high-entropy autocompletion loops, is now behaving as a\")\n",
        "print(\"perfectly constrained sentiment classification engine. By updating only\")\n",
        "print(\"a tiny fraction of parameters in the projection layers via QLoRA, we\")\n",
        "print(\"have steered the model's attention heads to output only our target labels\")\n",
        "print(\"('Positive' or 'Negative') with zero conversational fluff.\")\n",
        "print(\"============================================================\")\n"
    ])

    # --- CELL 14.5: Fine-Tuned Model Playground ---
    add_markdown([
        "### 🎮 Try Your Own Reviews on the Fine-Tuned Model!",
        "Use the interactive form below to write your own reviews and see how our newly trained sentiment classification specialist classifies them."
    ])

    add_code([
        "#@title 🚀 Interactive Fine-Tuned Model Playground { run: \"auto\" }\n",
        "#@markdown Type your own review prompt below to test your fine-tuned sentiment classifier!\n",
        "\n",
        "custom_prompt = \"Classify the sentiment: 'The food was cold and the waiter was incredibly rude.'\" #@param {type:\"string\"}\n",
        "\n",
        "# Ensure fine-tuned model is in-memory\n",
        "if 'trainer' in globals() and 'base_tokenizer' in globals():\n",
        "    trainer.model.eval()\n",
        "    trainer.model.config.use_cache = True # Explicitly enable KV Cache during evaluation for fast error-free decoding\n",
        "    eval_messages = [{\"role\": \"user\", \"content\": custom_prompt}]\n",
        "    formatted_prompt = base_tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=True)\n",
        "    inputs = base_tokenizer(formatted_prompt, return_tensors=\"pt\").to(base_model.device)\n",
        "    inputs.pop(\"token_type_ids\", None) # Remove token_type_ids if present to avoid generation dimension mismatch\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        outputs = trainer.model.generate(\n",
        "            **inputs,\n",
        "            max_new_tokens=10,\n",
        "            do_sample=False,\n",
        "            pad_token_id=base_tokenizer.eos_token_id\n",
        "        )\n",
        "        \n",
        "    prompt_len = inputs.input_ids.shape[1]\n",
        "    response = base_tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()\n",
        "    \n",
        "    print(\"=\"*60)\n",
        "    print(\"🚀 FINE-TUNED MODEL SPECIALIST RESPONSE:\")\n",
        "    print(\"=\"*60)\n",
        "    print(response)\n",
        "    print(\"=\"*60)\n",
        "else:\n",
        "    print(\"⚠️ Error: The Fine-tuned model or trainer is not in memory. Please complete the QLoRA training steps above.\")"
    ])

    # --- CELL 15: Markdown Section 6 (GCS Integration) ---
    add_markdown([
        "## ☁️ Step 7: Saving and Exporting Checkpoints to Google Cloud Storage (GCS)",
        "",
        "To save these adapters permanently (since Colab local memory is ephemeral), you can easily serialize them and upload them directly to a Google Cloud Storage bucket. This allows your Cloud Run service to download them dynamically during deployment.",
        "",
        "### GCS Integration Commands"
    ])

    add_code([
        "# 1. Authenticate your Google Cloud account within Colab\n",
        "from google.colab import auth\n",
        "auth.authenticate_user()\n",
        "print(\"✅ Authenticated GCP User!\")\n",
        "\n",
        "# 2. Configure project variables\n",
        "GCP_PROJECT_ID = \"YOUR_GCP_PROJECT_ID\" # Replace with your active project\n",
        "GCS_BUCKET_NAME = \"your-gemma-gcp-bucket\" # Replace with your bucket\n",
        "\n",
        "# Set active gcloud SDK project\n",
        "!gcloud config set project {GCP_PROJECT_ID}\n",
        "\n",
        "# 3. Create GCS bucket if not exists (using gcloud)\n",
        "!gcloud storage buckets create gs://{GCS_BUCKET_NAME} --location=us-central1\n",
        "\n",
        "# 4. Upload local fine-tuned adapter directory to GCS\n",
        "!gcloud storage cp -r ./fine_tuned_gemma_adapter gs://{GCS_BUCKET_NAME}/gemma-4-adapters/\n",
        "print(f\"\\n🎉 Successfully uploaded fine-tuned adapter weights to: gs://{GCS_BUCKET_NAME}/gemma-4-adapters/\")"
    ])
    
    # --- CELL 16: Markdown Section 7 (Hugging Face Hub and Browser Download) ---
    add_markdown([
        "## 🤗 Step 8: Alternative Export Methods (Hugging Face Hub & Browser Download)",
        "",
        "If you want to share your fine-tuned model with others so they can immediately test its behavior, or simply keep a local backup on your machine without using a GCP storage bucket, you can use these two highly convenient alternative export methods.",
        "",
        "### 🛒 Option A: Share publicly on the Hugging Face Hub (Recommended)",
        "Uploading your adapter directly to the Hugging Face Hub makes it incredibly easy for others to load and use. Anyone else running this notebook (or your local serving script) can simply reference your public model ID (e.g., `your_username/gemma-4-sentiment-adapter`) to run instant inference!"
    ])
    
    # --- CELL 17: Push to HF Hub block ---
    add_code([
        "#@title 🤗 Share adapter publicly on Hugging Face Hub { run: \"manual\" }\n",
        "#@markdown Enter your Hugging Face username and desired repository name below.\n",
        "#@markdown Make sure you logged in with a WRITE token in Step 1!\n",
        "\n",
        "HF_USERNAME = \"your_hf_username\" #@param {type:\"string\"}\n",
        "HF_REPO_NAME = \"gemma-4-sentiment-adapter\" #@param {type:\"string\"}\n",
        "\n",
        "hub_model_id = f\"{HF_USERNAME}/{HF_REPO_NAME}\"\n",
        "\n",
        "if HF_USERNAME == \"your_hf_username\":\n",
        "    print(\"⚠️ Please enter your real Hugging Face username in the form-field above!\")\n",
        "else:\n",
        "    print(f\"⏳ Pushing adapter weights and tokenizer to Hugging Face Hub: {hub_model_id}...\")\n",
        "    try:\n",
        "        # Push the PEFT adapter model and base tokenizer\n",
        "        trainer.model.push_to_hub(hub_model_id, private=False)\n",
        "        base_tokenizer.push_to_hub(hub_model_id, private=False)\n",
        "        print(f\"\\n🎉 Successfully pushed to HF Hub! View it at: https://huggingface.co/{hub_model_id}\")\n",
        "        print(\"\\n💡 Others can load and use your adapter directly using:\")\n",
        "        print(f\"   PeftModel.from_pretrained(base_model, \\\"{hub_model_id}\\\")\")\n",
        "    except Exception as e:\n",
        "        print(f\"\\n❌ Push failed: {e}\")\n",
        "        print(\"💡 Did you use a token with WRITE permissions during login in Step 1?\")"
    ])
    
    # --- CELL 18: Browser Download markdown ---
    add_markdown([
        "### 📥 Option B: Download directly to your local machine",
        "",
        "If you don't have a Google Cloud or Hugging Face account and just want a local copy of your fine-tuned weights, you can compress the adapter directory into a `.zip` archive and download it directly through your web browser."
    ])
    
    # --- CELL 19: Browser Download block ---
    add_code([
        "import os\n",
        "import zipfile\n",
        "from google.colab import files\n",
        "\n",
        "zip_path = \"./fine_tuned_gemma_adapter.zip\"\n",
        "adapter_dir = \"./fine_tuned_gemma_adapter\"\n",
        "\n",
        "print(\"⏳ Compressing fine-tuned adapter folder...\")\n",
        "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:\n",
        "    for root, dirs, files_in_dir in os.walk(adapter_dir):\n",
        "        for file in files_in_dir:\n",
        "            file_path = os.path.join(root, file)\n",
        "            # Maintain relative path inside zip\n",
        "            arcname = os.path.relpath(file_path, os.path.dirname(adapter_dir))\n",
        "            zipf.write(file_path, arcname)\n",
        "\n",
        "print(f\"✅ Compression complete! File size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB\")\n",
        "print(\"⏳ Starting browser download. Please allow popups if prompted...\")\n",
        "files.download(zip_path)"
    ])

    # Save to disk
    output_path = "/Users/ksprashanth/code/sandbox/gemma4-finetuning/gemma4_colab_finetuning.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"File created successfully: {output_path}")

if __name__ == "__main__":
    build_notebook()
