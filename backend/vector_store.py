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
    """
    本地兜底 Embedding。
    作用：当 HuggingFace 本地模型加载失败时，保证系统还能运行。
    注意：这个只适合开发测试，不适合正式语义检索。
    """

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
    """
    获取 Embedding 模型。
    优先加载 HuggingFace 本地模型；
    如果失败，则回退到 LocalHashEmbeddings。
    """

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={
                "device": "cpu",
                "local_files_only": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    except Exception as e:
        print(f"[Embedding Warning] HuggingFace Embedding 加载失败，使用 LocalHashEmbeddings。原因：{e}")
        return LocalHashEmbeddings()


def get_vector_store():
    """
    加载 FAISS 向量库。
    如果索引不存在，返回 None。
    """

    faiss_dir = Path(settings.vector_store_dir)

    if not (faiss_dir / "index.faiss").exists():
        return None

    return FAISS.load_local(
        folder_path=str(faiss_dir),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def save_chunks_to_faiss(chunks: list[str], metadatas: list[dict]):
    """
    保存文本切片到 FAISS。
    如果已有索引，则追加；
    如果没有索引，则新建。
    """

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
    """
    兼容旧调用。
    等价于 get_vector_store()。
    """

    return get_vector_store()


def reset_faiss_index():
    """
    清空 FAISS 索引目录。
    """

    faiss_dir = Path(settings.vector_store_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    for item in faiss_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def retrieve_top_k(question: str, top_k: int = 5):
    """
    Top-K 检索，用于前端可视化展示。
    """

    vector_store = get_vector_store()

    if vector_store is None:
        return []

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k,
    )

    formatted_results = []

    for index, (doc, score) in enumerate(results, start=1):
        formatted_results.append(
            {
                "rank": index,
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "未知来源"),
            }
        )

    return formatted_results