import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()

def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings