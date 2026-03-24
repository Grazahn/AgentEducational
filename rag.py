"""
Modul RAG - Retrieval-Augmented Generation.

Lant LCEL pentru raspunsuri la intrebari bazate exclusiv pe contextul
din materialul de curs. Reduce halucinatiile prin constrangeri stricte
in prompt.
"""

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from config import LLM_MODEL, LLM_TEMPERATURE_RAG
from utils import format_docs


_PROMPT_RAG: str = (
    "Esti un asistent strict si precis pentru studentii de la ETTI. "
    "Mai jos primesti un CONTEXT extras din curs si o INTREBARE.\n\n"
    "INFORMATII DOCUMENT (din titlul fisierului PDF, daca nu apar in context):\n"
    "- Nume materie / prescurtare curs: {nume_materie}\n\n"
    "REGULI:\n"
    "1. Raspunde FOLOSIND informatiile din CONTEXT sau din INFORMATII DOCUMENT (nume materie).\n"
    "2. Pentru intrebari despre NUMELE CURSului - foloseste numele din titlul fisierului daca nu e in context.\n"
    "3. Pentru punctaj, notare, laborator, seminar - cauta in context (deseori la inceput, sectiunea Punctaje).\n"
    "4. Spune 'Nu am gasit informatia in curs.' DOAR daca nu exista nici in context, nici in informatiile document.\n"
    "5. Nu oferi sfaturi generale. Citeaza cifre si date exacte din text cand exista.\n\n"
    "CONTEXT:\n{context}\n\n"
    "INTREBARE: {question}\n\n"
    "RASPUNS:"
)


def _extrage_nume_materie_din_pdf(cale_pdf: str) -> str:
    """Extrage numele materiei din titlul fisierului PDF (fara extensie)."""
    nume_fisier = os.path.basename(cale_pdf)
    nume, _ = os.path.splitext(nume_fisier)
    if not nume:
        return "necunoscut"
    return nume.replace("_", " ")


def creeaza_rag_chain(retriever: BaseRetriever, cale_pdf: str) -> RunnableSequence:
    """
    Construieste lantul LCEL pentru Q&A.

    Flux: intrebare -> retriever -> format_docs -> prompt -> LLM -> parser
    Numele materiei este extras din titlul PDF si adaugat in context.

    Args:
        retriever: Retriever Chroma pentru documente relevante
        cale_pdf: Calea la fisierul PDF (pentru a extrage numele materiei)

    Returns:
        Lant LCEL invocabil cu .invoke(intrebare)
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE_RAG)
    nume_materie = _extrage_nume_materie_din_pdf(cale_pdf)
    prompt = ChatPromptTemplate.from_template(_PROMPT_RAG).partial(
        nume_materie=nume_materie
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
