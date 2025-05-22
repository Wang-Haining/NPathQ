"""NPathQ: A Neuropathology Report QA Agent

This offline Gradio application lets you chat with the contents of a neuropathology
report PDF using any locally available LLM (default: Llama‑3.1‑8B‑Instruct).

Workflow
--------
1. Upload a PDF.
2. The file is parsed to plain text by `docling`, a unified document understanding
   framework from the Linux Foundation AI & Data ecosystem. It supports page layout,
   OCR, and multiple export formats.
3. Text is split into *overlapping* chunks (ca. 1k tokens). The overlap prevents
   fuzzy cut‑offs from hiding facts that straddle chunk boundaries. Each
   chunk is embedded with *all‑MiniLM‑L6‑v2*.
4. Chunks are stored in a local FAISS index.
5. A `ConversationalRetrievalChain` feeds the most relevant chunks plus the
   **system prompt** (loaded from *system_prompt.md*) into your chosen LLM.
   The LLM call uses the model's specific chat template via its Hugging Face tokenizer.
   Earlier turns are appended for on‑session memory until the PDF changes.
6. Answers stream to the chat UI.
"""

import argparse
from pathlib import Path
from typing import Any, ClassVar, Dict, List

import gradio as gr
import torch
from docling.document_converter import DocumentConverter
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import StringPromptTemplate
from pydantic import PrivateAttr
from transformers import AutoTokenizer

AGENT_NAME = "NPathQ"

LLM = None
TOKENIZER = None
SYSTEM_PROMPT = None
MODEL_ID = None


def _device() -> str:
    return (
        "cuda:1"
        if torch.cuda.device_count() > 1
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def load_llm_and_tokenizer(model_id: str, max_new: int = 1024):
    llm = VLLM(
        model=model_id,
        max_new_tokens=max_new,
        trust_remote_code=True,
        tensor_parallel_size=2 if '70b' in model_id.lower() else 1,
        enforce_eager=True if '70b' in model_id.lower() else False,
        dtype="bfloat16",
        streaming=True,
        temperature=0.7,
        top_p=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print(
            f"WARNING: Tokenizer for {model_id} does not have a chat_template defined "
            f"in its config. The model may not behave as expected. Falling back to "
            f"basic concatenation."
        )
    return llm, tokenizer


def pdf_to_text(pdf_path: Path) -> str:
    return DocumentConverter().convert(str(pdf_path)).document.export_to_text()


def vector_store(text: str):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=100
    ).create_documents([text])

    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": _device()},
    )
    return FAISS.from_documents(chunks, embed)


class ChatTemplatePrompt(StringPromptTemplate):
    """Wrap HF `tokenizer.chat_template` so LangChain can inject vars."""

    input_variables: ClassVar[List[str]] = ["context", "question", "chat_history"]
    _system_prompt: str = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, system_prompt: str, tokenizer, **kwargs):
        super().__init__(
            input_variables=self.input_variables, **kwargs
        )
        self._system_prompt = system_prompt
        self._tokenizer = tokenizer

    def format(self, **kwargs) -> str:
        ctx = kwargs["context"]
        query = kwargs["question"]

        chat_history_input = kwargs.get("chat_history") # Get it without default to see if None

        # debug
        print(f"DEBUG ChatTemplatePrompt: received chat_history_input is of type: {type(chat_history_input)}")
        if isinstance(chat_history_input, str):
            print(f"DEBUG ChatTemplatePrompt: chat_history_input (string sample): \"{chat_history_input[:100]}...\"")
        elif isinstance(chat_history_input, list) and chat_history_input:
            print(f"DEBUG ChatTemplatePrompt: chat_history_input (list, first item type): {type(chat_history_input[0])}")

        messages = [{"role": "system", "content": self._system_prompt}]

        # process chat_history_input carefully
        if isinstance(chat_history_input, list):
            for item in chat_history_input:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    user_msg, ai_msg = item
                    messages.append({"role": "user", "content": str(user_msg)})
                    messages.append({"role": "assistant", "content": str(ai_msg)})
                else:
                    print(f"WARNING ChatTemplatePrompt: Skipping malformed item in chat_history list: {item}")
        elif isinstance(chat_history_input, str) and chat_history_input.strip():
            print("WARNING ChatTemplatePrompt: chat_history_input was a string. "
                  "It will not be included in the structured 'messages' list for apply_chat_template. "
                  "Conversational context from history might be lost for the LLM.")


        # add current user query with context
        messages.append(
            {
                "role": "user",
                "content": f"Based on the following context:\n\n{ctx}\n\n"
                           f"Answer this question:\n{query}",
            }
        )

        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @property
    def _prompt_type(self) -> str:
        """Required by Langchain for custom prompts."""
        return "chat-template-prompt"


def qa_chain(vstore, system_prompt_content: str, llm_instance, tokenizer_instance):
    prompt = ChatTemplatePrompt(system_prompt_content, tokenizer_instance)

    return ConversationalRetrievalChain.from_llm(
        llm=llm_instance,
        retriever=vstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
    )


def upload_pdf(pdf_file_obj: gr.File, state: Dict[str, Any]):
    global LLM, TOKENIZER, SYSTEM_PROMPT
    if pdf_file_obj is None:
        raise gr.Error("Please upload a PDF first.")
    if LLM is None or TOKENIZER is None or SYSTEM_PROMPT is None:
        missing = []
        if LLM is None:
            missing.append("LLM")
        if TOKENIZER is None:
            missing.append("Tokenizer")
        if SYSTEM_PROMPT is None:
            missing.append("System Prompt")
        raise gr.Error(
            f"{', '.join(missing)} not initialized. Please ensure system_prompt.md exists and restart the application."
        )

    txt = pdf_to_text(Path(pdf_file_obj.name))
    vstore = vector_store(txt)

    state["chain"] = qa_chain(vstore, SYSTEM_PROMPT, LLM, TOKENIZER)
    state["langchain_history"] = []
    ui_message = [{"role": "system", "content": "PDF parsed. Ask away!"}]
    state["ui_messages"] = ui_message

    return (
        ui_message,
        state,
        gr.update(interactive=True, placeholder="Type your question and press Enter…"),
    )


def answer(msg: str, state: Dict[str, Any]):
    if not msg.strip():
        yield state.get("ui_messages", []), state
        return

    chain = state.get("chain")
    if not chain:
        current_ui_messages = state.get("ui_messages", []).copy()
        # avoid adding multiple parsing messages
        if (
                not current_ui_messages
                or current_ui_messages[-1].get("content")
                != "🔄 Still parsing the PDF or PDF not yet uploaded. Please wait or upload a PDF."
        ):
            current_ui_messages.append(
                {
                    "role": "assistant",
                    "content": "🔄 Still parsing the PDF or PDF not yet uploaded. Please wait or upload a PDF.",
                }
            )
        yield current_ui_messages, state
        return

    ui_messages = state.get("ui_messages", []).copy()
    langchain_history = state.get("langchain_history", []).copy()

    ui_messages.append({"role": "user", "content": msg})
    # add a placeholder for the assistant's response while processing
    ui_messages.append({"role": "assistant", "content": "🤔 Thinking..."}) # You can customize this message

    # update state with new user message and thinking indicator before yielding
    # this ensures the UI reflects the change immediately
    # state["ui_messages"] = ui_messages # Not strictly necessary to update state here if yield returns it
    yield ui_messages, state

    current_assistant_response = ""
    try:
        response_payload = chain.invoke(
            {"question": msg, "chat_history": langchain_history}
        )

        # conversationalRetrievalChain typically returns a dict with an "answer" key.
        current_assistant_response = response_payload.get("answer")

        if current_assistant_response is None:
            # fallback if 'answer' key is missing or the payload is different
            if isinstance(response_payload, str):
                current_assistant_response = response_payload
            elif isinstance(response_payload, dict) and not response_payload: # Empty dict
                current_assistant_response = "Received an empty response from the assistant."
            else:
                error_detail = f"Error: Could not extract answer. Response payload: {str(response_payload)[:200]}"
                print(f"Unexpected response payload structure: {response_payload}")
                current_assistant_response = error_detail

    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        current_assistant_response = "Sorry, an error occurred while generating the response."

    # update the last message in ui_messages (the assistant's placeholder) with the actual response
    if ui_messages and ui_messages[-1]["role"] == "assistant":
        ui_messages[-1]["content"] = current_assistant_response
    else:
        # This case should ideally not be reached if the placeholder was added correctly
        ui_messages.append({"role": "assistant", "content": current_assistant_response})

    # ensure current_assistant_response is a string for history storage
    if not isinstance(current_assistant_response, str):
        current_assistant_response = str(current_assistant_response)

    state["langchain_history"].append((msg, current_assistant_response))
    state["ui_messages"] = ui_messages # Final update to state

    yield ui_messages, state


def reset_session(state: Dict[str, Any]):
    initial_ui_messages = [
        {"role": "system", "content": "Session reset. Please upload a PDF to begin."}
    ]
    state["chain"] = None
    state["langchain_history"] = []
    state["ui_messages"] = initial_ui_messages
    return initial_ui_messages, state


CUSTOM_CSS = """
body {font-family: 'Inter', sans-serif;}
/* Titles */
#main-title-md h1 {text-align: center !important; font-size: 2.8em !important; margin-bottom: 0.1em !important; color: #333 !important;}
#sub-title-md p {text-align: center !important; font-size: 1.3em !important; color: #555 !important; margin-top: 0 !important; margin-bottom: 2em !important;}
/* Chat alignment */
.gradio-chatbot > .wrap {display: flex; flex-direction: column;}
.gradio-chatbot .message-wrap[data-testid="user"] {align-self: flex-start !important;}
.gradio-chatbot .message-wrap[data-testid="user"] > div.message {background-color: #DCF8C6 !important; color: #000 !important; max-width: 70% !important; border-radius: 10px !important;}
.gradio-chatbot .message-wrap[data-testid="assistant"] {align-self: flex-end !important;}
.gradio-chatbot .message-wrap[data-testid="assistant"] > div.message {background-color: #ECE5DD !important; color: #000 !important; max-width: 70% !important; border-radius: 10px !important;}
/* Footer */
#footer-info-md p {font-size: 0.8em !important; color: #888 !important; text-align: center !important; margin-top: 25px !important; padding: 15px !important; border-top: 1px solid #eee !important;}
#footer-info-md a {color: #007bff !important; text-decoration: none !important;}
#footer-info-md a:hover {text-decoration: underline !important;}
/* Device/Model info */
#device-model-info-md p {font-size: 0.75em !important; color: #aaa !important; text-align: center !important; margin-top: 5px !important;}
"""

FOOTER_TEXT = (
    f"Brought to you by Haining Wang@Su's Lab, "
    "Department of Biostatistics & Health Data Science, IU School of Medicine. "
    "Contact <a href='mailto:hw56@iu.edu'>hw56@iu.edu</a> for assistance."
)


def build_ui(port: int):
    global MODEL_ID
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown(f"# 🧠 {AGENT_NAME}", elem_id="main-title-md")
        gr.Markdown("The First Neuropathology Report QA Agent", elem_id="sub-title-md")
        app_state = gr.State(
            {
                "chain": None,
                "langchain_history": [],
                "ui_messages": [
                    {"role": "system", "content": "Please upload a PDF to begin."}
                ],
            }
        )
        with gr.Row():
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            reset_btn = gr.Button("Reset Chat & PDF")
        chat = gr.Chatbot(
            value=[{"role": "system", "content": "Please upload a PDF to begin."}],
            label=f"{AGENT_NAME} Chat",
            height=480,
            show_copy_button=True,
            layout="panel",
            type="messages",
        )
        box = gr.Textbox(
            lines=1,
            placeholder="Upload a PDF to begin asking questions.",
            label="Your Question",
            scale=7,
            interactive=False,
        )
        pdf_file.change(
            upload_pdf,
            inputs=[pdf_file, app_state],
            outputs=[chat, app_state, box],
        )

        reset_btn.click(
            reset_session, inputs=[app_state], outputs=[chat, app_state]
        ).then(
            lambda: gr.update(
                interactive=False, placeholder="Upload a PDF to begin asking questions."
            ),
            inputs=None,
            outputs=[box],
        )

        box.submit(fn=answer, inputs=[box, app_state], outputs=[chat, app_state])
        box.submit(
            fn=lambda: "", inputs=None, outputs=[box], queue=False
        )
        gr.Markdown(FOOTER_TEXT, elem_id="footer-info-md")
        gr.Markdown(
            f"<small>Device: {_device()} | LLM: {MODEL_ID}</small>",
            elem_id="device-model-info-md",
        )
    print(f"Starting {AGENT_NAME} on http://0.0.0.0:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{AGENT_NAME}: PDF QA with Local LLMs"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model ID for the LLM.",
    )
    parser.add_argument(
        "--prompt",
        default="system_prompt.md",
        help="Path to the system prompt Markdown file. This file MUST exist.",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the Gradio app on."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for LLM generation.",
    )
    args = parser.parse_args()

    MODEL_ID = args.model

    system_prompt_path = Path(args.prompt)
    if not system_prompt_path.exists():
        print(f"ERROR: System prompt file '{system_prompt_path}' not found.")
        print(
            "Please create this file with your desired system prompt or check the path."
        )
        exit(1)

    try:
        SYSTEM_PROMPT = system_prompt_path.read_text().strip()
        if not SYSTEM_PROMPT:
            print(f"ERROR: System prompt file '{system_prompt_path}' is empty.")
            print("Please ensure the file contains a valid system prompt.")
            exit(1)
        print(f"Successfully loaded system prompt from '{system_prompt_path}'.")
    except Exception as e:
        print(f"ERROR: Could not read system prompt file '{system_prompt_path}': {e}")
        exit(1)

    print(f"Loading LLM ({MODEL_ID}) and Tokenizer...")
    LLM, TOKENIZER = load_llm_and_tokenizer(MODEL_ID, max_new=args.max_new_tokens)
    print("LLM and Tokenizer loaded.")

    if LLM is None or TOKENIZER is None or SYSTEM_PROMPT is None:
        missing_init = []
        if LLM is None:
            missing_init.append("LLM")
        if TOKENIZER is None:
            missing_init.append("Tokenizer")
        if SYSTEM_PROMPT is None:
            missing_init.append("SYSTEM_PROMPT text")
        raise RuntimeError(
            f"Critical components not initialized before UI build: {', '.join(missing_init)}. Exiting."
        )

    build_ui(args.port)
