"""
Script de debug pentru a identifica de ce agentul nu gaseste informatia despre punctaj.
Ruleaza: python debug_retrieval.py
"""

import sys

# Config trebuie incarcat primul (seteaza env vars)
from config import CALE_PDF_IMPLICITA, CHUNK_SIZE, CHUNK_OVERLAP

# Fix encoding pe Windows (cp1252 nu suporta toate caracterele)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import incarca_sau_creaza_vector_store
from utils import format_docs


def main():
    print()
    print("DEBUG: De ce nu gaseste punctajul?")
    print()

    # 1. Ce extrage PyPDFLoader din primele pagini?
    print("\n1. EXTRAGERE PDF (primele 3 pagini)")
    loader = PyPDFLoader(CALE_PDF_IMPLICITA)
    docs = loader.load()
    for i, doc in enumerate(docs[:3]):
        text = doc.page_content.strip()
        print(f"\n>>> Pagina {i + 1} (len={len(text)} caractere):")
        text_preview = text[:800] if len(text) > 800 else text
        # Evita erori encoding pe Windows (ligaturi unicode etc.)
        text_preview = text_preview.encode("ascii", errors="replace").decode("ascii")
        print(text_preview)
        if len(text) > 800:
            print("... [trunchiat]")

    # 2. Cum arata chunk-urile dupa split?
    print("\n2. CHUNK-URI (primele 3)")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks[:3]):
        meta = c.metadata
        text = c.page_content[:500].encode("ascii", errors="replace").decode("ascii")
        print(f"\n>>> Chunk {i + 1} (pagina {meta.get('page', '?')}, len={len(c.page_content)}):")
        print(text + ("..." if len(c.page_content) > 500 else ""))
        # Cauta cuvinte cheie
        if any(w in c.page_content.lower() for w in ["punctaj", "laborator", "seminar", "curs", "examen"]):
            print("  >>> CONTINE cuvinte legate de punctaj!")
        print()

    # 3. Ce returneaza retriever-ul pentru intrebarea ta?
    print("\n3. CE GASESTE RETRIEVER-UL?")
    print("Intrebare: 'punctaj puncte laborator seminar curs'")
    print()
    vector_store, retriever = incarca_sau_creaza_vector_store(CALE_PDF_IMPLICITA)
    rezultate = retriever.invoke("punctaj puncte laborator seminar curs")
    print(f"Am gasit {len(rezultate)} chunk-uri.")
    for i, r in enumerate(rezultate):
        meta = r.metadata
        print(f"\nChunk returnat #{i + 1} (pagina {meta.get('page', '?')})")
        content = r.page_content[:600].encode("ascii", errors="replace").decode("ascii")
        print(content + ("..." if len(r.page_content) > 600 else ""))
        if any(w in r.page_content.lower() for w in ["punctaj", "laborator", "seminar", "curs"]):
            print("  >>> CONTINE info despre punctaj!")
    print("\n")


if __name__ == "__main__":
    main()
