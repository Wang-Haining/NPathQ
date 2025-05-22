# 🧠 NPathQ: A Neuropathology Report QA Agent

**NPathQ** is a privacy-first question answering agent built for analyzing neuropathology reports (PDFs). 
It features an LLM backend, multi-round conversation with memory, and document parsing tools from 
the [Docling](https://github.com/docling-project/docling) framework.

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/Wang-Haining/NPathQ.git
cd NPathQ
```

### 2. Create a virtual environment (recommended)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

We provide a fully frozen `requirements.txt` to ensure reproducibility:

```bash
python -m pip install -r requirements.txt
```

---

## ⚙️ Runtime Requirements

- Python 3.10+
- CUDA-enabled GPU (for vLLM + embedding)
- ThinLinc or SSH access to a compute host (optional)
- HuggingFace access token (if downloading Llama-3.1 or other gated models)

You must also have access to the model you're specifying (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`). 
To use private models, set the `HF_TOKEN` environment variable or log in via `huggingface-cli login`.

---

## 🧪 Running the App

To run the QA agent locally:

```bash
python app.py
```

The app will launch on `http://localhost:7860` by default. 
You can upload a neuropathology PDF and start querying in natural language.

---

## 📝 Customizing System Prompt

Edit `system_prompt.md` to define how the LLM should behave. 
A strong default is already included, which encourages faithful, evidence-grounded answers.

---

## 💬 Example Queries

After uploading a report, you can ask questions like:

- "What is the weight of the brain?"
- "Is there evidence of cerebral infarction?"
- "Were there any signs of hippocampal sclerosis?"
- "Was asymmetry noted between the frontal lobes?"

---


## 🧱 Project Structure

```
.
├── app.py                 # Main Gradio app entry point
├── system_prompt.md       # Custom system prompt for LLM
├── requirements.txt       # Frozen pip dependencies
└── README.md              # This file
```
---

## 🧑‍💻 Credits

NPathQ was developed by **Haining Wang** at the **Su Lab**, Indiana University School of Medicine.  
Contact: [hw56@iu.edu](mailto:hw56@iu.edu)


---

## 📄 License

MIT License (or replace with your institution’s license as applicable)

## Version

- May 21, 2025, v0.0.1: debut  
