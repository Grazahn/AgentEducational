"""
Modele Pydantic pentru output structurat.

Definesc schema pentru raspunsurile LLM-ului in modul Quiz,
permituind validare automata si parsare robusta.
"""

from pydantic import BaseModel, Field


class IntrebareGrila(BaseModel):
    """
    O intrebare tip grila cu variante de raspuns.

    Atribute:
        intrebare: Textul intrebarii
        variante: Lista de 4 optiuni (A, B, C, D)
        raspuns_corect: Indexul variantei corecte (0-3)
    """

    intrebare: str = Field(description="Textul intrebarii")
    variante: list[str] = Field(description="Lista de variante (4 optiuni: A, B, C, D)")
    raspuns_corect: int = Field(
        description="Indexul variantei corecte (0=A, 1=B, 2=C, 3=D)",
        ge=0,
        le=3,
    )


class QuizComplet(BaseModel):
    """
    Un quiz format din mai multe intrebari grila.

    Atribute:
        intrebari: Lista de IntrebareGrila generate din context
    """

    intrebari: list[IntrebareGrila] = Field(
        description="Lista de intrebari tip grila generate din context"
    )
