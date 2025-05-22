"""NPathQ: A Neuropathology Report QA Agent

This offline Gradio application lets you chat with the contents of a neuropathology
report PDF using any locally available LLM (default: Llama‑3.1‑8B‑Instruct).

Workflow
-------------------
1. Upload a PDF.
2. The file is parsed to plain text by `docling`.
3. The full PDF text is stored.
4. When a user asks a question:
   a. A prompt is constructed containing:
      i. The system prompt (from *system_prompt.md*).
      ii. The full text of the PDF.
      iii. The recent conversation history (up to 50 rounds).
      iv. The user's current question.
   b. This prompt is formatted using the model's chat template and sent to the LLM.
5. Answers are shown in the chat UI (non-streaming).
6. A warning is issued if the conversation length approaches 40 rounds.
"""

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import torch
from docling.document_converter import DocumentConverter
from langchain_community.llms import VLLM
from transformers import AutoTokenizer

AGENT_NAME = "NPathQ"

# globals for LLM instance, tokenizer, system prompt, and model ID
LLM = None
TOKENIZER = None
SYSTEM_PROMPT = None
MODEL_ID = None

# conversation limits
MAX_CONVERSATION_ROUNDS = 50
WARNING_CONVERSATION_ROUNDS = 40


def _device() -> str:
    # this function can indicate general GPU availability if VLLM is using it.
    return (
        "cuda:1"
        if torch.cuda.device_count() > 1
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def load_llm_and_tokenizer(model_id: str, max_new: int = 2048, cli_args=None):
    """
    Loads the main VLLM instance and the tokenizer.
    Ensures a chat template is available.
    """
    print("Loading VLLM and tokenizer...")
    is_70b = "70b" in model_id.lower()
    tensor_parallel_size = 2 if is_70b else 1
    enforce_eager = True if is_70b else False

    llm = VLLM(
        model=model_id,
        max_new_tokens=max_new,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        temperature=(
            cli_args.temperature
            if cli_args and hasattr(cli_args, "temperature")
            else 0.7
        ),
        top_p=cli_args.top_p if cli_args and hasattr(cli_args, "top_p") else 0.95,
    )
    print("VLLM loaded.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.chat_template:
        print(f"Tokenizer for {model_id} loaded with its own chat_template from config.")
    elif hasattr(tokenizer, 'default_chat_template') and tokenizer.default_chat_template:
        print(f"WARNING: Tokenizer for {model_id} did not have a chat_template in its config. "
              f"Using its `default_chat_template` provided by the transformers library.")
        tokenizer.chat_template = tokenizer.default_chat_template
    else:
        raise ValueError('Template Error.')

    print("Tokenizer loaded and chat template handling complete.")
    return llm, tokenizer


def pdf_to_text(pdf_path: Path) -> str:
    print(f"Parsing PDF: {pdf_path}")
    try:
        text = DocumentConverter().convert(str(pdf_path)).document.export_to_text()
        print(f"PDF parsed successfully. Text length: {len(text)} characters.")
        return text
    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")
        print(traceback.format_exc())
        raise gr.Error(f"Failed to parse PDF: {e}")


def format_llm_prompt(
    system_prompt_str: str,
    pdf_content_str: str,
    history_tuples: List[Tuple[str, str]],
    current_question_str: str,
    tokenizer_instance: Any,
) -> str:
    """
    Formats the prompt for the LLM using the tokenizer's chat template.
    The PDF content and conversation history are embedded within a single user message.
    """
    messages = [{"role": "system", "content": system_prompt_str}]

    user_message_content_parts = []

    # 1. PDF Content
    user_message_content_parts.append(
        f"**Full Neuropathology Report Content:**\n{pdf_content_str}"
    )

    # 2. conversation History (if any)
    # use only the last MAX_CONVERSATION_ROUNDS from history_tuples
    if history_tuples:
        relevant_history = history_tuples[-MAX_CONVERSATION_ROUNDS:]
        history_block_parts = []
        for user_msg, assistant_msg in relevant_history:
            history_block_parts.append(f"User: {user_msg}")
            history_block_parts.append(f"Assistant: {assistant_msg}")
        user_message_content_parts.append(
            f"**Conversation History (most recent {len(relevant_history)} turns):**\n{'\n'.join(history_block_parts)}"
        )

    # 3. current Question
    user_message_content_parts.append(
        f"**Current Question (based on the report and history):**\n{current_question_str}"
    )

    full_user_content = "\n\n---\n\n".join(user_message_content_parts)
    messages.append({"role": "user", "content": full_user_content})

    try:
        formatted_prompt = tokenizer_instance.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        # fallback for very basic tokenizers if template is broken or missing despite checks
        if messages[0]['role'] == 'system':
            formatted_prompt = messages[0]['content'] + "\n" + messages[1]['content']
        else:
            formatted_prompt = messages[0]['content']
        print("Warning: Using basic concatenation for prompt due to template error.")


    print("\n---- DEBUG: format_llm_prompt - PROMPT TO TOKENIZER (STRUCTURE) ----")
    print(f"System Prompt: {messages[0]['content'][:200]}...")
    if len(messages) > 1:
         print(f"User Content (start): {messages[1]['content'][:300]}...")
         print(f"User Content (end): ...{messages[1]['content'][-300:]}")
    print("--- Generated Prompt String (first 300 chars) ---")
    print(formatted_prompt[:300])
    print(f"--- Generated Prompt String Length: {len(formatted_prompt)} ---")
    print("------------------------------------------------------------------\n")
    return formatted_prompt


def upload_pdf(pdf_file_obj: gr.File, state: Dict[str, Any]):
    global LLM, TOKENIZER, SYSTEM_PROMPT
    if pdf_file_obj is None:
        raise gr.Error("Please upload a PDF first.")

    missing = []
    if LLM is None: missing.append("LLM")
    if TOKENIZER is None: missing.append("Tokenizer")
    if SYSTEM_PROMPT is None: missing.append("System Prompt")
    if missing:
        raise gr.Error(
            f"{', '.join(missing)} not initialized. Please ensure system_prompt.md exists and restart."
        )

    txt = pdf_to_text(Path(pdf_file_obj.name))
    state["pdf_text"] = txt
    state["conversation_history"] = [] # Reset history for the new PDF
    ui_message = [{"role": "system", "content": "PDF parsed. You can now ask questions about its content."}]
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

    pdf_text = state.get("pdf_text")
    if not pdf_text:
        current_ui_messages = state.get("ui_messages", []).copy()
        if not current_ui_messages or current_ui_messages[-1].get("content") != "Please upload a PDF first.":
            current_ui_messages.append(
                {"role": "assistant", "content": "Please upload a PDF first."}
            )
        yield current_ui_messages, state
        return

    ui_messages = state.get("ui_messages", []).copy()
    conversation_history_for_prompt = state.get("conversation_history", []).copy()

    ui_messages.append({"role": "user", "content": msg})
    ui_messages.append({"role": "assistant", "content": "🤔 Thinking..."})
    yield ui_messages, state

    llm_response_text = ""
    try:
        prompt_str = format_llm_prompt(
            SYSTEM_PROMPT,
            pdf_text,
            conversation_history_for_prompt,
            msg,
            TOKENIZER,
        )
        # VLLM's invoke method directly takes the prompt string
        llm_response_text = LLM.invoke(prompt_str)
        if not llm_response_text or not llm_response_text.strip():
            llm_response_text = "I received an empty response from the model. Please try rephrasing your question."

    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        print(traceback.format_exc())
        llm_response_text = "Sorry, an error occurred while generating the response. Please check the console for details."

    # add conversation length warning if needed
    warning_message = ""
    num_rounds_completed = len(conversation_history_for_prompt) # Rounds *before* this one

    if num_rounds_completed + 1 >= MAX_CONVERSATION_ROUNDS:
        warning_message = (
            f"\n\n---\n**SYSTEM WARNING:** Conversation has reached {num_rounds_completed + 1} rounds "
            f"(max {MAX_CONVERSATION_ROUNDS}). To ensure optimal performance and context accuracy, "
            "please **reset the chat and PDF** to start a new session for further questions."
        )
    elif num_rounds_completed + 1 >= WARNING_CONVERSATION_ROUNDS:
        warning_message = (
            f"\n\n---\n**SYSTEM INFO:** Conversation at {num_rounds_completed + 1} rounds. "
            f"The context includes up to {MAX_CONVERSATION_ROUNDS} rounds. For very long discussions, "
            "consider starting a new session by resetting the chat and PDF."
        )

    final_assistant_response = llm_response_text.strip() + warning_message

    if ui_messages and ui_messages[-1]["role"] == "assistant":
        ui_messages[-1]["content"] = final_assistant_response
    else: # Should not happen if "Thinking..." was appended
        ui_messages.append({"role": "assistant", "content": final_assistant_response})

    # Update history *after* getting the response for *this* turn
    state["conversation_history"].append((msg, llm_response_text.strip())) # Store raw response
    state["ui_messages"] = ui_messages
    yield ui_messages, state


def reset_session(state: Dict[str, Any]):
    initial_ui_messages = [
        {"role": "system", "content": "Session reset. Please upload a PDF to begin."}
    ]
    state["pdf_text"] = None
    state["conversation_history"] = []
    state["ui_messages"] = initial_ui_messages
    # also reset the textbox
    return initial_ui_messages, state, gr.update(value="", placeholder="Upload a PDF to begin asking questions.", interactive=False)


CUSTOM_CSS = """
body {font-family: 'Inter', sans-serif;}
#main-title-md h1 {text-align: center !important; font-size: 2.8em !important; margin-bottom: 0.1em !important; color: #333 !important;}
#sub-title-md p {text-align: center !important; font-size: 1.3em !important; color: #555 !important; margin-top: 0 !important; margin-bottom: 2em !important;}
.gradio-chatbot > .wrap {display: flex; flex-direction: column;}
.gradio-chatbot .message-wrap[data-testid="user"] {align-self: flex-start !important;}
.gradio-chatbot .message-wrap[data-testid="user"] > div.message {background-color: #DCF8C6 !important; color: #000 !important; max-width: 70% !important; border-radius: 10px !important;}
.gradio-chatbot .message-wrap[data-testid="assistant"] {align-self: flex-end !important;}
.gradio-chatbot .message-wrap[data-testid="assistant"] > div.message {background-color: #ECE5DD !important; color: #000 !important; max-width: 70% !important; border-radius: 10px !important;}
#footer-info-md p {font-size: 0.8em !important; color: #888 !important; text-align: center !important; margin-top: 25px !important; padding: 15px !important; border-top: 1px solid #eee !important;}
#footer-info-md a {color: #007bff !important; text-decoration: none !important;}
#footer-info-md a:hover {text-decoration: underline !important;}
#device-model-info-md p {font-size: 0.75em !important; color: #aaa !important; text-align: center !important; margin-top: 5px !important;}
"""

FOOTER_TEXT = (
    f"Brought to you by Haining Wang@Su's Lab, "
    "Department of Biostatistics & Health Data Science, IU School of Medicine. "
    "Contact <a href='mailto:hw56@iu.edu'>hw56@iu.edu</a> for assistance."
)


def build_ui(port: int, share_the_ui: bool):
    global MODEL_ID
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown(f"# 🧠 {AGENT_NAME}", elem_id="main-title-md")
        gr.Markdown("Simplified Neuropathology Report QA Agent", elem_id="sub-title-md") # Updated subtitle

        # initial state values
        initial_ui_messages_val = [{"role": "system", "content": "Please upload a PDF to begin."}]
        app_state = gr.State(
            {
                "pdf_text": None,
                "conversation_history": [],
                "ui_messages": initial_ui_messages_val,
            }
        )
        with gr.Row():
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            reset_btn = gr.Button("Reset Chat & PDF")

        chat = gr.Chatbot(
            value=initial_ui_messages_val,
            label=f"{AGENT_NAME} Chat",
            height=500, #
            show_copy_button=True,
            bubble_full_width=False, # For better message alignment
            layout="panel",
        )


        box = gr.Textbox(
            lines=1,
            placeholder="Upload a PDF to begin asking questions.",
            label="Your Question",
            scale=7, #
            interactive=False,
        )
        pdf_file.change(
            upload_pdf,
            inputs=[pdf_file, app_state],
            outputs=[chat, app_state, box],
        )

        reset_btn.click(
            reset_session,
            inputs=[app_state],
            outputs=[chat, app_state, box] # reset_session now also outputs for the textbox
        )
        # .then removed as reset_session now handles the textbox update.

        box.submit(fn=answer, inputs=[box, app_state], outputs=[chat, app_state])
        box.submit(fn=lambda: "", inputs=None, outputs=[box], queue=False) # Clears textbox after submit

        gr.Markdown(FOOTER_TEXT, elem_id="footer-info-md")
        gr.Markdown(
            f"<small>Device: {_device()} | LLM: {MODEL_ID} | Max Rounds: {MAX_CONVERSATION_ROUNDS}</small>",
            elem_id="device-model-info-md",
        )
    print(f"Starting {AGENT_NAME} on http://0.0.0.0:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share_the_ui)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{AGENT_NAME}: Simplified PDF QA with Local LLMs"
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
        default=4096, # Increased from original 2048, good for comprehensive answers
        help="Max new tokens for the main LLM generation.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for the LLM." # Slightly lower default
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top_p for the LLM.") # Slightly lower default
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable external access to the app via public Gradio link.",
    )
    args = parser.parse_args()

    MODEL_ID = args.model

    system_prompt_path = Path(args.prompt)
    if not system_prompt_path.exists():
        print(f"ERROR: System prompt file '{system_prompt_path}' not found. Please create it.")
        print("Example system_prompt.md content:")
        print("---")
        print("You are NPathQ, a specialized AI assistant for Neuropathology Report Question Answering.")
        print("Carefully analyze the provided neuropathology report content and the conversation history to answer the user's questions accurately and concisely.")
        print("If the information is not in the report, state that explicitly.")
        print("---")
        exit(1)
    try:
        SYSTEM_PROMPT = system_prompt_path.read_text().strip()
        if not SYSTEM_PROMPT:
            print(f"ERROR: System prompt file '{system_prompt_path}' is empty.")
            exit(1)
        print(f"Successfully loaded system prompt from '{system_prompt_path}'.")
    except Exception as e:
        print(f"ERROR: Could not read system prompt file '{system_prompt_path}': {e}")
        exit(1)

    # load single LLM and tokenizer
    LLM, TOKENIZER = load_llm_and_tokenizer(
        MODEL_ID, max_new=args.max_new_tokens, cli_args=args
    )

    if LLM is None or TOKENIZER is None or SYSTEM_PROMPT is None:
        missing_init = []
        if LLM is None: missing_init.append("LLM")
        if TOKENIZER is None: missing_init.append("Tokenizer")
        if SYSTEM_PROMPT is None: missing_init.append("SYSTEM_PROMPT text") # Should be caught by earlier checks
        # This specific error condition might be redundant due to earlier exits, but good as a safeguard.
        raise RuntimeError(
            f"Critical components not initialized before UI build: {', '.join(missing_init)}. Exiting."
        )

    build_ui(args.port, share_the_ui=args.share)