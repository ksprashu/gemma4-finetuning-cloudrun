# Agentic Dataset Synthesis with Google Antigravity (AGY)

When preparing custom training datasets for model fine-tuning, standard programmatic generators use template strings with rigid structures and pre-defined lists of terms. While helpful for basic validation, programmatic datasets often lack the linguistic richness, vocabulary diversity, and subtle edge cases necessary to train a highly robust production model.

To build production-grade classifiers or instruction-tuned task specialists, you need **high-fidelity synthetic data**. 

By leveraging **Google Antigravity (AGY)**, you can programmatically and agentically prompt the AI to act as a **Producer Archetype** to generate thousands of unique, natural, and complex samples conforming to strict JSON schema structures.

---

## 🚀 Why Use Google Antigravity (AGY)?

1. **Semantic Diversity**: Rather than repeating static templates, the Producer Archetype writes unique conversational inputs with varying lengths, tones, and cultural context.
2. **Edge-Case Simulation**: Direct the generator to synthesize complex linguistic phenomena, such as double negations, sarcasm, mixed sentiments, slang, typos, and multilingual inputs.
3. **Structured Outputs**: Ensure every generated sample perfectly matches Gemma's conversational training schema (`{"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}`) and can be read directly by the Hugging Face `datasets` library.
4. **Auto-Persistence**: AGY automatically executes, verifies, and saves the output directly as a `.jsonl` file in your workspace, making it immediately available for upload to Google Cloud Storage.

---

## 🛠️ Setup and Install AGY

Ensure you have AGY running on your local machine:
* **Antigravity CLI**: Run `agy` in your terminal to start an interactive agentic session. Authenticate with Google on the first run.
* **Antigravity 2.0 (UI)**: Launch the parallel Desktop Client. Open your project folder to grant workspace access.

---

## 🎭 Enterprise Fine-Tuning Scenarios & Prompts

Here are four high-value domain-specific use cases you can fine-tune Gemma 4 on, along with the precise prompt you can copy and paste directly into AGY:

### 1. Sarcastic Sentiment & Emotion Classifier

* **Goal**: Train a highly sensitive classifier that goes beyond simple positive/negative to capture nuanced emotions, sarcasm, and passive-aggressive tones.
* **Resulting Dataset**: `sentiment_dataset.jsonl`

**AGY Prompt:**
```text
Role: Act as a data generation engineer (Producer Archetype).
Task: Create a synthetic fine-tuning dataset of 1000 highly varied sentiment and emotional classification samples.
Constraints:
- Out of these, 400 must contain subtle sarcasm, 200 mixed-sentiments, 200 double-negations, and 200 realistic typos/slang.
- Target sentiment classes: "Positive", "Negative", "Sarcastic", "Frustrated".
- Format: Save the output directly as a JSON Lines (.jsonl) file in my current workspace named `sentiment_dataset.jsonl`.
- Schema: Every line must be a single JSON object matching this structure exactly:
  {"messages": [{"role": "user", "content": "Classify the sentiment: '<REVIEW_TEXT>'"}, {"role": "model", "content": "<LABEL>"}]}

Begin generating now, ensuring diverse vocabulary across sectors (food, apps, hotels, tech gadgets). Avoid repeating review structures.
```

---

### 2. Multilingual Support Ticket Router

* **Goal**: Train Gemma to behave as an intelligent IT support gateway that parses customer tickets in multiple languages and outputs a structured triage payload.
* **Resulting Dataset**: `triage_dataset.jsonl`

**AGY Prompt:**
```text
Role: Act as a data generation engineer (Producer Archetype).
Task: Create a synthetic fine-tuning dataset of 1000 customer support triage samples.
Constraints:
- The input review should be in random languages (English, Spanish, French, German, Japanese, Hindi).
- The model must output a JSON block indicating category and priority.
- Target Categories: "billing", "technical_support", "account_security", "refund_request".
- Target Priorities: "critical", "high", "medium", "low".
- Format: Save as a JSON Lines (.jsonl) file in my current workspace named `triage_dataset.jsonl`.
- Schema: Every line must be a single JSON object matching this structure exactly:
  {"messages": [{"role": "user", "content": "Triage this support ticket: '<TICKET_TEXT>'"}, {"role": "model", "content": "{\\\"category\\\": \\\"<CAT>\\\", \\\"priority\\\": \\\"<PRIORITY>\\\"}"}]}

Begin generating now, ensuring highly realistic support scenarios (e.g. payment failure, locked out of account, api latency, pricing query).
```

---

### 3. Text-to-API Payload Copilot

* **Goal**: Fine-tune Gemma to act as a natural language function caller, mapping spoken intents into structured REST API JSON payloads for backend services.
* **Resulting Dataset**: `api_dataset.jsonl`

**AGY Prompt:**
```text
Role: Act as a data generation engineer (Producer Archetype).
Task: Create a synthetic fine-tuning dataset of 1000 Text-to-API function call translation samples.
Constraints:
- The user prompt should be a natural language request to perform an action (e.g., book a flight, update a user profile, send a notification).
- The model response should be a formatted, structured REST API payload.
- Format: Save as a JSON Lines (.jsonl) file in my current workspace named `api_dataset.jsonl`.
- Schema: Every line must be a single JSON object matching this structure exactly:
  {"messages": [{"role": "user", "content": "Translate to API payload: '<REQUEST_TEXT>'"}, {"role": "model", "content": "{\\\"action\\\": \\\"<ACTION>\\\", \\\"params\\\": {<PARAMETERS>}}"}]}

Ensure wide variety of operations (Create, Read, Update, Delete) across CRM, booking, and notification system paradigms.
```

---

### 4. Enterprise PII Redactor

* **Goal**: Train Gemma to process raw customer service chats and automatically mask/redact sensitive Personally Identifiable Information (PII) before logging.
* **Resulting Dataset**: `pii_dataset.jsonl`

**AGY Prompt:**
```text
Role: Act as a data generation engineer (Producer Archetype).
Task: Create a synthetic fine-tuning dataset of 1000 PII anonymization samples.
Constraints:
- The input should be a conversational transcript containing names, credit card numbers, email addresses, phone numbers, or SSNs.
- The model response must be the exact same transcript, but with all PII replaced by standardized tags like `[REDACTED_NAME]`, `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_CARD]`.
- Format: Save as a JSON Lines (.jsonl) file in my current workspace named `pii_dataset.jsonl`.
- Schema: Every line must be a single JSON object matching this structure exactly:
  {"messages": [{"role": "user", "content": "Redact PII from this transcript: '<TRANSCRIPT>'"}, {"role": "model", "content": "<REDACTED_TRANSCRIPT>"}]}

Ensure diverse realistic chat snippets with conversational natural flow, typos, and varied customer service contexts.
```

---

## 🏃 Execution & Next Steps

1. Launch `agy` in your terminal or open the Antigravity 2.0 Chat.
2. Copy and paste any of the prompts above into the session.
3. Antigravity will automatically coordinate the **Producer Archetype** to write and structure the dataset.
4. Once completed, a `.jsonl` file with the requested name will be successfully created in your local project root.
5. You can now upload this dataset to Google Cloud Storage using:
   ```bash
   gcloud storage cp your_dataset.jsonl gs://your-gemma-gcp-bucket/datasets/your_dataset.jsonl
   ```
