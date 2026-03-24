"""
Agent Virtual Educational - ETTI

Pipeline RAG (Retrieval-Augmented Generation) + Modul Quiz pentru
asistarea studentilor. Rulare 100% locala, offline.

Tech stack: PyTorch (embeddings), LangChain (LCEL), ChromaDB, Ollama.
"""

# Config trebuie incarcat primul (seteaza env vars pentru HuggingFace)
from config import CALE_PDF_IMPLICITA
from vector_store import incarca_sau_creaza_vector_store, get_retriever_pentru_quiz
from rag import creeaza_rag_chain
from quiz import creeaza_quiz_chain, afiseaza_quiz


def _rulare_rag(vector_store, retriever, rag_chain) -> None:
    """Flux interactiv pentru intrebari RAG."""
    intrebare = input("\nIntrebarea ta: ").strip()
    if not intrebare:
        return
    print("\nSe proceseaza...")
    raspuns = rag_chain.invoke(intrebare)
    print("\nRaspuns:")
    print(raspuns)


def _rulare_quiz(vector_store, retriever) -> None:
    """Flux interactiv pentru generare quiz."""
    tema = input(
        "\nTema pentru quiz (ex: reguli de curs, laborator): "
    ).strip()
    if not tema:
        return
    retriever_quiz = get_retriever_pentru_quiz(vector_store)
    quiz_chain = creeaza_quiz_chain(retriever_quiz)
    print("\nSe genereaza intrebarile... (poate dura 30-60 secunde)")
    quiz = quiz_chain.invoke(tema)
    print("\nQuiz generat:")
    afiseaza_quiz(quiz)


def _recreare_baza(vector_store, retriever, rag_chain) -> tuple:
    """Recreeaza baza vectoriala si returneaza noile obiecte."""
    cale = input(f"\nCale PDF [{CALE_PDF_IMPLICITA}]: ").strip() or CALE_PDF_IMPLICITA
    print("\nSe recreeaza baza... (poate dura 1-2 minute)")
    vector_store, retriever = incarca_sau_creaza_vector_store(
        cale, forta_recreare=True, vector_store_vechi=vector_store
    )
    rag_chain = creeaza_rag_chain(retriever, cale)
    print("Baza vectoriala recreata. Poți continua cu intrebari si quiz.")
    return vector_store, retriever, rag_chain


def main() -> None:
    """Punct de intrare principal. Meniu interactiv."""
    print()
    print("  Agent Virtual Educational - ETTI")
    print("  RAG + Quiz | PyTorch + LangChain | Local & Offline")
    print()

    vector_store, retriever = incarca_sau_creaza_vector_store(CALE_PDF_IMPLICITA)
    rag_chain = creeaza_rag_chain(retriever, CALE_PDF_IMPLICITA)

    while True:
        print("\nMeniu:")
        print("1) Intreaba agentul (RAG)")
        print("2) Genereaza quiz pe o tema")
        print("3) Recreaza baza vectoriala (PDF nou sau reindexare)")
        print("4) Iesire")
        optiune = input("\nAlege (1/2/3/4): ").strip()

        if optiune == "1":
            _rulare_rag(vector_store, retriever, rag_chain)
        elif optiune == "2":
            _rulare_quiz(vector_store, retriever)
        elif optiune == "3":
            vector_store, retriever, rag_chain = _recreare_baza(
                vector_store, retriever, rag_chain
            )
        elif optiune == "4":
            print("La revedere!")
            break
        else:
            print("Optiune invalida. Alege 1, 2, 3 sau 4.")


if __name__ == "__main__":
    main()
