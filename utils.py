"""
Utilitare comune pentru pipeline-ul RAG si modulul Quiz.
"""

from typing import Sequence

from langchain_core.documents import Document


def format_docs(docs: Sequence[Document]) -> str:
    """
    Concateneaza continutul documentelor intr-un singur bloc de text.

    Folosit ca input pentru prompt-urile RAG si Quiz.

    Args:
        docs: Lista de documente LangChain

    Returns:
        Text concatenat, separate prin linii duble
    """
    return "\n\n".join(doc.page_content for doc in docs)
