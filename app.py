"""
NPathQ: A Neuropathology Report QA Agent
====================================================

This **fully offline** Gradio application lets you chat with the contents of a
PDF using any locally available causal‑LM (default: *Meta‑Llama‑3.1‑8B‑Instruct*).
It is aimed at GPU workstations/clusters where privacy or air‑gap constraints
rule out cloud APIs.

Workflow
--------
1. **Upload a PDF**.
2. The file is parsed to plain text by `docling`, a unified document understanding
   framework from the Linux Foundation AI & Data ecosystem. It supports page layout,
   OCR, and multiple export formats.
3. Text is split into *overlapping* chunks (≈1k tokens) – the overlap prevents
   fuzzy cut‑offs from hiding facts that straddle chunk boundaries – and each
   chunk is embedded with *all‑MiniLM‑L6‑v2*.
4. Chunks are stored in a local FAISS index.
5. A `ConversationalRetrievalChain` feeds the most relevant chunks plus the
   **system prompt** (loaded from *system_prompt.md*) into your chosen LLM.
   The LLM call uses the model's specific chat template via its Hugging Face tokenizer.
   Earlier turns are appended for on‑session memory until the PDF changes.
6. Answers stream to the chat UI. A static footer provides attribution.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import gradio as gr
import torch
from docling.document_converter import DocumentConverter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer

AGENT_NAME = "NPathQ"

LLM = None
TOKENIZER = None
SYSTEM_PROMPT = None
MODEL_ID = None


# ------------------------------ helpers -----------------------------------
def _device() -> str:
    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"

def load_llm_and_tokenizer(model_id: str, max_new: int = 1024):
    llm = VLLM(
        model=model_id,
        max_new_tokens=max_new,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        streaming=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print(f"Warning: Tokenizer for {model_id} does not have a chat_template defined. Defaulting to basic concatenation which might be suboptimal.")
    return llm, tokenizer

def pdf_to_text(pdf_path: Path) -> str:
    return DocumentConverter().convert(str(pdf_path)).document.export_to_text()

def vector_store(text: str):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    ).create_documents([text])

    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": _device()},
    )
    return FAISS.from_documents(chunks, embed)


def qa_chain(vstore, system_prompt_content: str, llm_instance, tokenizer_instance):
    messages_for_template = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": "Based on the following context:\n\n{context}\n\nAnswer this question:\n{question}"}
    ]

    try:
        prompt_template_str = tokenizer_instance.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}. Falling back to basic prompt string.")
        prompt_template_str = f"{system_prompt_content}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}\n\nAnswer:"

    final_prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_instance,
        retriever=vstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain_kwargs={"prompt": final_prompt},
        return_source_documents=False
    )
    return chain

# ------------------------------ gradio callbacks --------------------------
def upload_pdf(pdf_file_obj: gr.File, state: Dict[str, Any]):
    global LLM, TOKENIZER, SYSTEM_PROMPT
    if pdf_file_obj is None:
        raise gr.Error("Please upload a PDF first.")
    if LLM is None or TOKENIZER is None or SYSTEM_PROMPT is None:
        raise gr.Error("LLM, Tokenizer, or System Prompt not initialized. Please restart.")

    txt = pdf_to_text(Path(pdf_file_obj.name))
    vstore = vector_store(txt)

    state["chain"] = qa_chain(vstore, SYSTEM_PROMPT, LLM, TOKENIZER)
    state["langchain_history"] = []
    initial_ui_message = [{"role": "system", "content": "PDF parsed. Ask away!"}]
    state["ui_messages"] = initial_ui_message

    return initial_ui_message, state

def answer(msg: str, state: Dict[str, Any]):
    if not msg.strip():
        yield state.get("ui_messages", []), state
        return

    if "chain" not in state or not state["chain"]:
        raise gr.Error("PDF not processed or chain not initialized. Please upload a PDF first.")

    ui_messages = state.get("ui_messages", []).copy()
    langchain_history = state.get("langchain_history", []).copy()

    ui_messages.append({"role": "user", "content": msg})
    yield ui_messages, state

    ui_messages.append({"role": "assistant", "content": ""})
    current_assistant_response = ""

    try:
        for chunk in state["chain"].stream({"question": msg, "chat_history": langchain_history}):
            if "answer" in chunk:
                token = chunk["answer"]
                current_assistant_response += token
                ui_messages[-1]["content"] = current_assistant_response
                yield ui_messages, state
    except Exception as e:
        print(f"Error during LLM streaming: {e}")
        ui_messages[-1]["content"] = "Sorry, an error occurred while generating the response."
        state["ui_messages"] = ui_messages
        yield ui_messages, state
        return

    ui_messages[-1]["content"] = current_assistant_response

    state["langchain_history"].append((msg, current_assistant_response))
    state["ui_messages"] = ui_messages

    yield ui_messages, state

def reset_session(state: Dict[str, Any]):
    initial_ui_messages = [{"role": "system", "content": "Session reset. Please upload a PDF to begin."}]
    state["chain"] = None
    state["langchain_history"] = []
    state["ui_messages"] = initial_ui_messages
    return initial_ui_messages, state

# ------------------------------ UI ----------------------------------------
CUSTOM_CSS = """
body {font-family: 'Inter', sans-serif;}

/* Titles */
#main-title-md h1 { /* Gradio typically wraps # markdown in <h1> */
    text-align: center !important;
    font-size: 2.8em !important; /* Increased size */
    margin-bottom: 0.1em !important; /* Reduced space to subtitle */
    color: #333 !important;
}
#sub-title-md p { /* Subtitle is plain markdown, likely <p> */
    text-align: center !important;
    font-size: 1.3em !important; /* Increased size */
    color: #555 !important;
    margin-top: 0 !important;
    margin-bottom: 2em !important; /* Space before chat controls */
}

/* Chat alignment */
.gradio-chatbot > .wrap { 
    display: flex;
    flex-direction: column;
}
.gradio-chatbot .message-wrap[data-testid="user"] {
    align-self: flex-start !important;
}
.gradio-chatbot .message-wrap[data-testid="user"] > div.message {
    background-color: #DCF8C6 !important; /* Light green for user */
    color: #000 !important;
    max-width: 70% !important;
    border-radius: 10px !important;
}
.gradio-chatbot .message-wrap[data-testid="assistant"] {
    align-self: flex-end !important;
}
.gradio-chatbot .message-wrap[data-testid="assistant"] > div.message {
    background-color: #ECE5DD !important; /* Light beige for assistant */
    color: #000 !important;
    max-width: 70% !important;
    border-radius: 10px !important;
}

/* Footer */
#footer-info-md p { /* Footer is plain markdown, likely <p> */
    font-size: 0.8em !important;
    color: #888 !important; /* Lighter gray for footer */
    text-align: center !important;
    margin-top: 25px !important;
    padding: 15px !important;
    border-top: 1px solid #eee !important;
}
#footer-info-md a { /* Style links in footer */
    color: #007bff !important;
    text-decoration: none !important;
}
#footer-info-md a:hover {
    text-decoration: underline !important;
}

/* Device/Model info */
#device-model-info-md p {
    font-size: 0.75em !important;
    color: #aaa !important;
    text-align: center !important;
    margin-top: 5px !important;
}
"""

FOOTER_TEXT = (
    f"Brought to you by Haining Wang@Su's Lab, "
    "Department of Biostatistics & Health Data Science, IU School of Medicine. "
    "Contact <a href='mailto:hw56@iu.edu'>hw56@iu.edu</a> for assistance."
)


def build_ui(port: int):
    global MODEL_ID
    with gr.Blocks(css=CUSTOM_CSS) as demo:

        gr.Markdown(f"# 🧠 {AGENT_NAME}", elem_id="main-title-md") # Main Title
        gr.Markdown("The First Neuropathology Report QA Agent", elem_id="sub-title-md") # Subtitle

        app_state = gr.State({
            "chain": None,
            "langchain_history": [],
            "ui_messages": [{"role": "system", "content": "Please upload a PDF to begin."}]
        })

        with gr.Row():
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            reset_btn = gr.Button("Reset Chat & PDF")

        chat = gr.Chatbot(
            value=[{"role": "system", "content": "Please upload a PDF to begin."}],
            label=f"{AGENT_NAME} Chat", # Label might not be visible if layout="panel" takes over
            height=480,
            show_copy_button=True,
            layout="panel",
            type="messages"
        )
        box  = gr.Textbox(lines=1, placeholder="Type your question and press Enter…", label="Your Question", scale=7)

        pdf_file.change(upload_pdf, inputs=[pdf_file, app_state], outputs=[chat, app_state])
        reset_btn.click(reset_session, inputs=[app_state], outputs=[chat, app_state])

        box.submit(fn=answer, inputs=[box, app_state], outputs=[chat, app_state])
        box.submit(fn=lambda: "", inputs=None, outputs=[box], queue=False)

        gr.Markdown(FOOTER_TEXT, elem_id="footer-info-md") # Static Footer
        gr.Markdown(f"<small>Device: {_device()} | LLM: {MODEL_ID}</small>", elem_id="device-model-info-md")


    print(f"Starting {AGENT_NAME} on http://0.0.0.0:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

# ------------------------------ main --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{AGENT_NAME}: PDF QA with Local LLMs")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face model ID for the LLM.")
    parser.add_argument("--prompt", default="system_prompt.md", help="Path to the system prompt Markdown file.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens for LLM generation.")
    args = parser.parse_args()

    MODEL_ID = args.model

    system_prompt_path = Path(args.prompt)
    if not system_prompt_path.exists():
        print(f"Error: System prompt file not found at {system_prompt_path}")
        default_prompt_content = "You are a helpful AI assistant specialized in answering questions about neuropathology reports. Use the provided context from the report to answer accurately and concisely. If the information is not in the context, say you don't know."
        try:
            with open(system_prompt_path, "w") as f:
                f.write(default_prompt_content)
            print(f"Created a default system prompt at: {system_prompt_path}")
            SYSTEM_PROMPT = default_prompt_content
        except Exception as e:
            print(f"Could not create default system prompt: {e}. Exiting.")
            exit(1)
    else:
        SYSTEM_PROMPT = system_prompt_path.read_text().strip()

    print(f"Loading LLM ({MODEL_ID}) and Tokenizer...")
    LLM, TOKENIZER = load_llm_and_tokenizer(MODEL_ID, max_new=args.max_new_tokens)
    print("LLM and Tokenizer loaded.")

    build_ui(args.port)
