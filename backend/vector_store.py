from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from .config import get_settings


settings = get_settings()


def get_embeddings():
    """
    使用本地多语言 embedding 模型。
    适合中英文企业知识库原型。
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def save_chunks_to_faiss(chunks: list[str], metadatas: list[dict]):
    """
    将切分后的文本 chunks 写入 FAISS。
    如果已有 FAISS 索引，则追加；
    如果没有，则新建。
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

    if faiss_dir.exists() and (faiss_dir / "index.faiss").exists():
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
    加载本地 FAISS 向量库。
    """
    embeddings = get_embeddings()
    faiss_dir = Path(settings.vector_store_dir)

    if not faiss_dir.exists() or not (faiss_dir / "index.faiss").exists():
        return None

    return FAISS.load_local(
        folder_path=str(faiss_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )