"""
Utilitare pentru randarea imaginilor de pagina din PDF.

Imaginile sunt generate doar pentru paginile relevante, astfel incat
retrieval-ul text ramane rapid, iar suportul multimodal apare doar cand ajuta.
"""

import os
from typing import Optional

import fitz

from config import FOLDER_ASSETS_DOCUMENTE, SCALA_IMAGINI_PDF


def _sanitize_name(name: str) -> str:
    """Creeaza un nume sigur pentru foldere si fisiere auxiliare."""
    chars = []
    for char in name:
        chars.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(chars).strip("_") or "document"


def get_assets_dir_for_pdf(cale_pdf: str) -> str:
    """Returneaza folderul in care se stocheaza imaginile paginilor PDF."""
    base_name = os.path.splitext(os.path.basename(cale_pdf))[0]
    return os.path.join(FOLDER_ASSETS_DOCUMENTE, _sanitize_name(base_name))


def get_page_image_path(cale_pdf: str, page_index: int) -> str:
    """Construieste calea standard pentru imaginea unei pagini din PDF."""
    return os.path.join(
        get_assets_dir_for_pdf(cale_pdf),
        f"page_{page_index + 1:03d}.png",
    )


def ensure_page_image(cale_pdf: str, page_index: int) -> Optional[str]:
    """
    Genereaza imaginea unei pagini daca lipseste si returneaza calea ei.

    Returneaza None daca pagina nu exista.
    """
    image_path = get_page_image_path(cale_pdf, page_index)
    if os.path.exists(image_path):
        return image_path

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    with fitz.open(cale_pdf) as pdf_document:
        if page_index < 0 or page_index >= pdf_document.page_count:
            return None

        page = pdf_document.load_page(page_index)
        matrix = fitz.Matrix(SCALA_IMAGINI_PDF, SCALA_IMAGINI_PDF)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(image_path)

    return image_path
