"""
Modul Quiz - Generare intrebari tip grila pe baza materialului de curs.

Lant LCEL cu output structurat (Pydantic) pentru formularea automata
a intrebarilor de evaluare.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.retrievers import BaseRetriever

from config import LLM_MODEL, LLM_TEMPERATURE_QUIZ, NUMAR_INTREBARI_QUIZ
from models import QuizComplet
from utils import format_docs


_PROMPT_QUIZ: str = (
    "Esti un profesor care creaza teste de evaluare pentru studentii de la ETTI. "
    "Mai jos ai un CONTEXT extras din materialul de curs.\n\n"
    "TEMA CERUTA: {topic}\n\n"
    "CONTEXT DIN CURS:\n{context}\n\n"
    "CERINTE:\n"
    "- Toate intrebarile si variantele de raspuns trebuie formulate STRICT in limba romana.\n"
    "- Genereaza exact {n} intrebari tip grila.\n"
    "- Fiecare intrebare are 4 variante (A, B, C, D), doar UNA corecta.\n"
    "- Raspunsul corect TREBUIE sa provina STRICT din contextul de mai sus.\n"
    "- Intrebarile sa fie relevante pentru tema ceruta.\n"
    "- variante: lista de 4 string-uri cu DOAR textul raspunsului, FARA prefixe A), B), C), D) la inceput.\n"
    "- raspuns_corect: indexul variantei corecte (0=A, 1=B, 2=C, 3=D).\n\n"
    "Genereaza quiz-ul in formatul structurat cerut, in limba romana."
)


def creeaza_quiz_chain(retriever: BaseRetriever) -> RunnableSequence:
    """
    Construieste lantul LCEL pentru generare quiz.

    Flux: tema -> retriever -> format_docs -> prompt -> LLM (structured) -> QuizComplet

    Args:
        retriever: Retriever cu k mare pentru context suficient

    Returns:
        Lant invocabil care returneaza obiect QuizComplet
    """
    llm_base = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE_QUIZ)
    llm = llm_base.with_structured_output(QuizComplet)
    prompt = ChatPromptTemplate.from_template(_PROMPT_QUIZ)

    return (
        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
        | prompt.partial(n=NUMAR_INTREBARI_QUIZ)
        | llm
    )


def _curata_varianta(text: str, index_litera: int) -> str:
    """
    Elimina prefix redundant (A), B), etc.) daca LLM l-a inclus din greseala.
    """
    litere = ["A", "B", "C", "D"]
    prefix = f"{litere[index_litera]}) "
    t = text.strip()
    if t.upper().startswith(prefix.upper()):
        return t[len(prefix) :].strip()
    return t


def afiseaza_quiz(quiz: QuizComplet) -> None:
    """
    Afiseaza quiz-ul formatat in consola.

    Raspunsul corect este marcat cu (*).
    """
    print(format_quiz_pentru_gui(quiz))


def format_quiz_pentru_gui(quiz: QuizComplet) -> str:
    """
    Returneaza quiz-ul formatat ca Markdown pentru afisare in GUI.

    Raspunsul corect este marcat cu (*).
    """
    litere = ["A", "B", "C", "D"]
    linii = []
    for i, q in enumerate(quiz.intrebari, 1):
        linii.append(f"### Intrebare {i}\n{q.intrebare}\n")
        for j, v in enumerate(q.variante):
            v_curat = _curata_varianta(v, j)
            marcaj = " **(*)**" if j == q.raspuns_corect else ""
            linii.append(f"- **{litere[j]})** {v_curat}{marcaj}")
        linii.append("")
    return "\n".join(linii)
