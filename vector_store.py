"""
Modul pentru gestionarea bazei de date vectoriale ChromaDB.

Foloseste embeddings bazate pe PyTorch (sentence-transformers) pentru
indexare semantica a documentelor. Modelul multilingual este optim pentru
text in limba romana.
"""

import gc
import os
import shutil
import time
from typing import Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from config import (
    FOLDER_DB_VECTORIALA,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    K_RAG,
    K_QUIZ,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returneaza instanta de embeddings configurata.

    Foloseste sentence-transformers (PyTorch) - ruleaza 100% local.

    Returns:
        HuggingFaceEmbeddings configurat pentru modelul multilingual
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def _inchide_client_chroma(vector_store) -> None:
    """Inchide conexiunea Chroma pentru a elibera fisierele (necesar pe Windows)."""
    if vector_store is None:
        return
    try:
        client = getattr(vector_store, "_client", None)
        if client is not None and hasattr(client, "close"):
            client.close()
    except Exception:
        pass


def incarca_sau_creaza_vector_store(
    cale_pdf: str,
    forta_recreare: bool = False,
    vector_store_vechi: Optional[VectorStore] = None,
) -> Tuple[VectorStore, BaseRetriever]:
    """
    Incarca baza de date vectoriala existenta sau o creeaza din PDF.

    La prima rulare sau la forta_recreare=True, PDF-ul este procesat,
    chunk-uit si indexat cu embeddings PyTorch.

    Args:
        cale_pdf: Calea catre fisierul PDF cu materialul de curs
        forta_recreare: Daca True, ignora DB existenta si recreeaza din PDF

    Returns:
        Tuple (vector_store, retriever) pentru utilizare in RAG si Quiz
    """
    embeddings = get_embeddings()

    if not forta_recreare and os.path.exists(FOLDER_DB_VECTORIALA):
        print("Baza de date vectoriala exista. Se incarca...")
        vector_store = Chroma(
            persist_directory=FOLDER_DB_VECTORIALA,
            embedding_function=embeddings,
        )
        print("Incarcare finalizata.")
    else:
        if os.path.exists(FOLDER_DB_VECTORIALA):
            _inchide_client_chroma(vector_store_vechi)
            gc.collect()
            time.sleep(0.5)
            print("Se sterge baza vectoriala veche...")
            try:
                shutil.rmtree(FOLDER_DB_VECTORIALA)
            except PermissionError:
                print(
                    "Eroare: Baza e blocata. Inchide programul, sterge manual folderul "
                    "'db_vectoriala', apoi ruleaza din nou si alege optiunea 3."
                )
                raise

        print("Extragere si Chunking PDF")
        loader = PyPDFLoader(cale_pdf)
        documente = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        bucati_text = splitter.split_documents(documente)
        print(f"Am impartit documentul in {len(bucati_text)} bucati.")

        print("\nCreare Baza de Date Vectoriala (embeddings PyTorch)")
        print("Se genereaza vectorii... (poate dura 1-2 minute)")
        vector_store = Chroma.from_documents(
            documents=bucati_text,
            embedding=embeddings,
            persist_directory=FOLDER_DB_VECTORIALA,
        )
        print("Baza de date creata cu succes.")

    retriever = vector_store.as_retriever(search_kwargs={"k": K_RAG})
    return vector_store, retriever


def get_retriever_pentru_quiz(
    vector_store: VectorStore,
    k: Optional[int] = None,
) -> BaseRetriever:
    """
    Returneaza un retriever cu configuratie optima pentru generare quiz.

    Quiz-ul beneficiaza de mai mult context pentru a formula intrebari coerente.

    Args:
        vector_store: Store-ul vectorial Chroma
        k: Numar chunk-uri (implicit din config)

    Returns:
        Retriever configurat pentru modul Quiz
    """
    num_chunks = k if k is not None else K_QUIZ
    return vector_store.as_retriever(search_kwargs={"k": num_chunks})
