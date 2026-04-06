"""
Interfata Grafika - Agent Virtual Educational ETTI

Frontend simplu cu Gradio: Chat RAG, Quiz, Recreare baza.
Ruleaza: python app_gradio.py
"""

import os
import time

# Config trebuie incarcat primul (seteaza env vars pentru HuggingFace)
from config import CALE_PDF_IMPLICITA
import gradio as gr
from vector_store import incarca_sau_creaza_vector_store, get_retriever_pentru_quiz
from rag import creeaza_rag_chain
from quiz import creeaza_quiz_chain, format_quiz_pentru_gui


def _init_state():
    """Incarca agentul la pornire."""
    cale = CALE_PDF_IMPLICITA
    vs, ret = incarca_sau_creaza_vector_store(cale)
    quiz_retriever = get_retriever_pentru_quiz(vs)
    chain = creeaza_rag_chain(ret, cale)
    quiz_chain = creeaza_quiz_chain(quiz_retriever)
    return {
        "vector_store": vs,
        "retriever": ret,
        "quiz_retriever": quiz_retriever,
        "rag_chain": chain,
        "quiz_chain": quiz_chain,
        "cale_pdf": cale,
    }


def _append_chat(history, user_msg: str, assistant_msg: str):
    """Adauga un schimb de replici in formatul Gradio 6 (messages)."""
    return history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def chat_fn(message, history, state):
    """Raspunde la intrebare folosind RAG."""
    if not state or state.get("rag_chain") is None:
        return _append_chat(history, message, "Eroare: Agentul nu e incarcat."), state
    try:
        start = time.perf_counter()
        raspuns = state["rag_chain"].invoke(message)
        print(f"[Timing] RAG total: {time.perf_counter() - start:.2f}s")
        return _append_chat(history, message, raspuns), state
    except Exception as e:
        return _append_chat(history, message, f"Eroare: {str(e)}"), state


def quiz_fn(tema, state):
    """Genereaza quiz pe tema data."""
    if not state or state.get("quiz_chain") is None:
        return "Eroare: Baza de date nu e incarcata.", state
    if not tema or not tema.strip():
        return "Introdu o tema pentru quiz.", state
    try:
        start = time.perf_counter()
        quiz = state["quiz_chain"].invoke(tema.strip())
        print(f"[Timing] Quiz total: {time.perf_counter() - start:.2f}s")
        return format_quiz_pentru_gui(quiz), state
    except Exception as e:
        return f"Eroare la generare: {str(e)}", state


def recreare_fn(cale, state):
    """Recreeaza baza vectoriala din PDF nou."""
    cale = (cale or "").strip() or CALE_PDF_IMPLICITA
    if not os.path.exists(cale):
        return f"Fisierul nu exista: {cale}", state
    try:
        vs_vechi = state.get("vector_store") if state else None
        vs, ret = incarca_sau_creaza_vector_store(
            cale, forta_recreare=True, vector_store_vechi=vs_vechi
        )
        chain = creeaza_rag_chain(ret, cale)
        quiz_retriever = get_retriever_pentru_quiz(vs)
        quiz_chain = creeaza_quiz_chain(quiz_retriever)
        new_state = {
            "vector_store": vs,
            "retriever": ret,
            "quiz_retriever": quiz_retriever,
            "rag_chain": chain,
            "quiz_chain": quiz_chain,
            "cale_pdf": cale,
        }
        return f"Baza recreata din {os.path.basename(cale)}. Poti continua.", new_state
    except Exception as e:
        return f"Eroare: {str(e)}", state


def build_ui():
    """Construieste interfata Gradio."""
    with gr.Blocks(title="Agent Educational ETTI") as demo:
        gr.Markdown("# Agent Virtual Educational - ETTI\n*RAG + Quiz | Local & Offline*")

        state = gr.State(_init_state)

        with gr.Tabs():
            # Tab 1: Chat
            with gr.TabItem("Intreaba agentul"):
                chatbot = gr.Chatbot(label="Conversatie", height=400)
                msg = gr.Textbox(
                    label="Intrebarea ta",
                    placeholder="Ex: Care sunt punctajele la laborator?",
                    show_label=False,
                    container=False,
                )
                with gr.Row():
                    submit_btn = gr.Button("Trimite", variant="primary")

                def user_submit(message, history, s):
                    if not message.strip():
                        return history, s
                    new_hist, new_s = chat_fn(message, history, s)
                    return new_hist, new_s

                msg.submit(user_submit, [msg, chatbot, state], [chatbot, state])
                submit_btn.click(user_submit, [msg, chatbot, state], [chatbot, state]).then(
                    lambda: "", None, msg
                )

            # Tab 2: Quiz
            with gr.TabItem("Genereaza quiz"):
                quiz_tema = gr.Textbox(
                    label="Tema pentru quiz",
                    placeholder="Ex: reguli de curs, laborator, generatii retele mobile",
                )
                quiz_btn = gr.Button("Genereaza intrebari", variant="primary")
                quiz_out = gr.Markdown(label="Quiz generat")

                quiz_btn.click(quiz_fn, [quiz_tema, state], [quiz_out, state])

            # Tab 3: Setari
            with gr.TabItem("Recreaza baza"):
                gr.Markdown("Incarca un PDF nou sau reindexeaza cursul curent.")
                cale_in = gr.Textbox(
                    label="Cale PDF",
                    value=CALE_PDF_IMPLICITA,
                    placeholder="cursuri_pdf/RCM.pdf",
                )
                recreare_btn = gr.Button("Recreeaza baza", variant="secondary")
                recreare_out = gr.Textbox(label="Status", interactive=False)

                recreare_btn.click(
                    recreare_fn,
                    [cale_in, state],
                    [recreare_out, state],
                )

        gr.Markdown("---\n*ETTI | PyTorch + LangChain + ChromaDB + LM Studio*")

    return demo


if __name__ == "__main__":
    print("Se incarca agentul... (poate dura cateva secunde)")
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css="footer {display: none !important}",
    )
