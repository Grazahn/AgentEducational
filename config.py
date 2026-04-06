"""
Configuratie centralizata pentru Agentul Virtual Educational.

Toate constantele aplicatiei sunt definite aici pentru mentenanta usoara
si configurare transparenta.
"""

import os
import logging
import warnings

# Setari de mediu - trebuie inainte de importul modelelor HuggingFace
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")


# CAI SI RESURSE


CALE_PDF_IMPLICITA: str = "cursuri_pdf/RCM.pdf"
FOLDER_DB_VECTORIALA: str = "./db_vectoriala"


# EMBEDDINGS - PyTorch / Sentence-Transformers
# Modelul multilingual functioneaza bine pentru text in romana.


EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE: str = "cpu"  # "cuda" daca ai GPU NVIDIA


# CHUNKING PDF


CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


# RETRIEVAL


K_RAG: int = 5  # numar chunk-uri pentru Q&A
K_QUIZ: int = 5  # numar chunk-uri pentru generare quiz


# LLM - LM Studio (endpoint OpenAI-compatible, local)


LLM_BASE_URL: str = "http://127.0.0.1:1234/v1"
LLM_API_KEY: str = "lm-studio"
LLM_MODEL: str = "qwen2.5-vl-7b-instruct"
#LLM_MODEL: str = "gemma-4-e4b-it"
LLM_TEMPERATURE_RAG: float = 0.0  # determinist pentru raspunsuri precise
LLM_TEMPERATURE_QUIZ: float = 0.3  # putina varietate pentru intrebari


# QUIZ


NUMAR_INTREBARI_QUIZ: int = 5
AFISEAZA_TIMPI: bool = True
