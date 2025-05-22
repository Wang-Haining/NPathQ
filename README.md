# 🧠 NPathQ: Neuropathology Report QA Made Easy & Private

**NPathQ** allows you chat directly with neuropathology report PDF using a locally hosted Large Language Model (LLM).
It features multi-round conversation with memory and good document parsing.

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/Wang-Haining/NPathQ.git
cd NPathQ
```

### 2. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
````

### 3. Install dependencies

We provide a `requirements.txt` file. It's recommended to freeze your environment for full reproducibility once stable.

```bash
python -m pip install -r requirements.txt
```

---

## ⚙️ Runtime Requirements

- Python 3.10+
- CUDA-enabled GPU with sufficient VRAM (for vLLM, especially important as the full PDF text and conversation history are loaded into context). The required VRAM will depend on the PDF size, chosen LLM, and conversation length.
- Access to a compute host with GPU (if not running locally).
- HuggingFace access token (if downloading Llama-3.1 or other gated models).

You must have access to the model you're specifying (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`).
To use private or gated models, set the `HF_TOKEN` environment variable or log in via `huggingface-cli login`.

---

## 🧪 Running the App

To run the QA agent locally:

```bash
python app.py --model <your_model_id> --prompt system_prompt.md
```

**Example:**
```bash
python app.py
```

The app will launch on `http://localhost:7860` by default.
You can upload a neuropathology PDF, and its entire text content will be used in the context for the LLM to answer your questions.

**Command-line arguments:**
-   `--model`: (Required if not default) Hugging Face model ID for the LLM (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`).
-   `--prompt`: Path to the system prompt Markdown file (default: `system_prompt.md`).
-   `--port`: Port to run the Gradio app on (default: `7860`).
-   `--max_new_tokens`: Max new tokens for LLM generation (default: `4096`).
-   `--temperature`: Temperature for the LLM (default: `0.6`).
-   `--top_p`: Top_p for the LLM (default: `0.9`).
-   `--share`: Enable external access via a public Gradio link.

---

## 📝 Customizing System Prompt

Edit `system_prompt.md` to define how the LLM should behave.
A strong default is already included, which encourages faithful, evidence-grounded answers based *solely* on the provided PDF content and conversation history.

---

## 💬 Example Queries

After uploading a report, you can ask questions like:

- "What is the weight of the brain according to the report?"
- "Is there evidence of cerebral infarction mentioned in this document?"
- "Does the report describe any signs of hippocampal sclerosis?"
- "Was asymmetry noted between the frontal lobes in this case?"

The LLM will answer based on the full text of the PDF you uploaded.

---

## ✨ Workflow

1.  **Upload PDF:** User uploads a neuropathology report.
2.  **Parse Text:** `docling` extracts the full plain text from the PDF.
3.  **Chat with LLM:**
    *   The complete PDF text, the system prompt, and the ongoing conversation history (up to a set limit) are formatted into a single prompt.
    *   This comprehensive prompt is sent to the locally running LLM (e.g., Llama-3.1-8B-Instruct via vLLM).
    *   The LLM generates a response based on all provided context.
4.  **Display Answer:** The LLM's answer is shown in the Gradio chat interface.
5.  **Conversation Limit:** The conversation history included in the prompt is limited (e.g., to 50 rounds) to manage context window size. Users are warned as they approach this limit.

---

## 🧱 Project Structure

```
.
├── app.py                 # Main Gradio app entry point
├── system_prompt.md       # Custom system prompt for the LLM
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🧑‍💻 Credits

NPathQ was developed by **Haining Wang** at the **Su Lab**, Department of Biostatistics & Health Data Science, Indiana University School of Medicine.
Contact: [hw56@iu.edu](mailto:hw56@iu.edu)

---

## 📄 License

MIT

---

## 🗓️ Version History

-   v0.1.0 (Current): *May 22, 2025* (or current date): Simplified architecture. Full PDF text loaded into LLM context. Removed vector store and retrieval chain. Added conversation limits and warnings.
-   v0.0.1: *May 21, 2025*: Initial release (with vector store and retrieval).

---