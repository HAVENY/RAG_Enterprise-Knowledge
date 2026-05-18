import hashlib
import re
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings


settings = get_settings()


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)

        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9_]+", text.lower())
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return words + cjk_chars


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return LocalHashEmbeddings()


def save_chunks_to_faiss(chunks: list[str], metadatas: list[dict]):
    embeddings = get_embeddings()

    docs = [
        LangChainDocument(
            page_content=chunk,
            metadata=metadata,
        )
        for chunk, metadata in zip(chunks, metadatas)
    ]

    faiss_dir = Path(settings.vector_store_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    if (faiss_dir / "index.faiss").exists():
        vector_store = FAISS.load_local(
            folder_path=str(faiss_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(docs)
    else:
        vector_store = FAISS.from_documents(
            documents=docs,
            embedding=embeddings,
        )

    vector_store.save_local(str(faiss_dir))


def load_faiss():
    faiss_dir = Path(settings.vector_store_dir)

    if not (faiss_dir / "index.faiss").exists():
        return None

    return FAISS.load_local(
        folder_path=str(faiss_dir),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def reset_faiss_index():
    faiss_dir = Path(settings.vector_store_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    for item in faiss_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
