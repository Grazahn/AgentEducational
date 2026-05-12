"""
Modul RAG - Retrieval-Augmented Generation.

Pastram retrieval-ul semantic pe text pentru viteza, dar putem trimite si
imagini relevante ale paginilor PDF catre un model multimodal din LM Studio.
"""

import base64
import os
import time
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever

from config import (
    ACTIVEAZA_RAG_MULTIMODAL,
    AFISEAZA_TIMPI,
    LLM_TEMPERATURE_RAG,
    MAX_IMAGINI_CONTEXT_RAG,
)
from llm_factory import creeaza_chat_llm, model_suporta_input_vizual
from pdf_assets import ensure_page_image
from utils import format_docs


_SYSTEM_PROMPT_RAG: str = (
    "Esti un asistent strict si precis pentru studentii de la ETTI. "
    "Raspunzi folosind doar contextul extras din curs si, daca sunt disponibile, "
    "imaginile paginilor relevante din acelasi curs. "
    "Daca informatia cautata apare intr-un grafic, tabel, schema sau figura, "
    "foloseste si continutul vizual pentru raspuns. "
    "Nu inventa detalii care nu apar in material."
)

_PROMPT_UTILIZATOR_RAG: str = (
    "INFORMATII DOCUMENT:\n"
    "- Nume materie / prescurtare curs: {nume_materie}\n\n"
    "REGULI:\n"
    "1. Raspunde folosind informatiile din CONTEXT si, daca sunt trimise, din IMAGINILE paginilor relevante.\n"
    "2. Pentru intrebari despre numele cursului, poti folosi numele extras din titlul PDF.\n"
    "3. Pentru punctaj, notare, laborator, seminar sau date exacte, prefera formularea fidela dupa curs.\n"
    "4. Daca nu gasesti informatia nici in context, nici in imaginile trimise, spune: 'Nu am gasit informatia in curs.'\n"
    "5. Daca raspunsul depinde de un tabel, grafic sau figura, mentioneaza pe scurt asta in raspuns.\n\n"
    "CONTEXT:\n{context}\n\n"
    "INTREBARE: {question}\n"
)


class MultimodalRAGChain:
    """Lant RAG custom care poate trimite si imagini paginilor relevante."""

    def __init__(self, retriever: BaseRetriever, cale_pdf: str) -> None:
        self._retriever = retriever
        self._cale_pdf = cale_pdf
        self._llm = creeaza_chat_llm(LLM_TEMPERATURE_RAG)
        self._nume_materie = _extrage_nume_materie_din_pdf(cale_pdf)
        self._poate_folosi_imagini = (
            ACTIVEAZA_RAG_MULTIMODAL and model_suporta_input_vizual()
        )

    def invoke(self, question: str) -> str:
        docs = self._retrage_documente(question)
        context = format_docs(docs)
        image_paths = self._selecteaza_imagini_relevante(docs)
        start = time.perf_counter()
        try:
            response = self._invoke_llm(question, context, image_paths)
        except Exception:
            if not image_paths:
                raise
            # Daca endpointul/modelul refuza imaginile, reluam text-only.
            response = self._invoke_llm(question, context, [])
        if AFISEAZA_TIMPI:
            print(f"[Timing] RAG generatie: {time.perf_counter() - start:.2f}s")
        return _extrage_text_din_raspuns(response)

    def _retrage_documente(self, question: str) -> Sequence[Document]:
        start = time.perf_counter()
        docs = self._retriever.invoke(question)
        if AFISEAZA_TIMPI:
            print(f"[Timing] RAG retrieval: {time.perf_counter() - start:.2f}s")
        return docs

    def _selecteaza_imagini_relevante(self, docs: Sequence[Document]) -> list[str]:
        if not self._poate_folosi_imagini:
            return []

        selected_paths: list[str] = []
        seen_pages: set[int] = set()

        for doc in docs:
            page_index = doc.metadata.get("page")
            if not isinstance(page_index, int) or page_index in seen_pages:
                continue
            image_path = ensure_page_image(self._cale_pdf, page_index)
            if image_path is None:
                continue
            selected_paths.append(image_path)
            seen_pages.add(page_index)
            if len(selected_paths) >= MAX_IMAGINI_CONTEXT_RAG:
                break

        if AFISEAZA_TIMPI and selected_paths:
            print(f"[Timing] Imagini context RAG: {len(selected_paths)}")
        return selected_paths

    def _invoke_llm(self, question: str, context: str, image_paths: Sequence[str]):
        user_content: list[dict] = [
            {
                "type": "text",
                "text": _PROMPT_UTILIZATOR_RAG.format(
                    nume_materie=self._nume_materie,
                    context=context,
                    question=question,
                ),
            }
        ]
        for image_path in image_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_path_to_data_url(image_path)},
                }
            )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT_RAG),
            HumanMessage(content=user_content),
        ]
        return self._llm.invoke(messages)


def _extrage_nume_materie_din_pdf(cale_pdf: str) -> str:
    """Extrage numele materiei din titlul fisierului PDF."""
    nume_fisier = os.path.basename(cale_pdf)
    nume, _ = os.path.splitext(nume_fisier)
    if not nume:
        return "necunoscut"
    return nume.replace("_", " ")


def _image_path_to_data_url(image_path: str) -> str:
    """Converteste imaginea locala intr-un data URL compatibil OpenAI-style."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extrage_text_din_raspuns(response) -> str:
    """Normalizeaza raspunsul LangChain intr-un string simplu."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(text for text in texts if text).strip()
    return str(content)


def creeaza_rag_chain(retriever: BaseRetriever, cale_pdf: str) -> MultimodalRAGChain:
    """Construieste lantul RAG text + imagini pentru paginile relevante."""
    return MultimodalRAGChain(retriever, cale_pdf)
