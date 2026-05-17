from pathlib import Path
from typing import List

from sqlalchemy.orm import Session
from langchain_core.documents import Document as LCDocument

from .models import Document
from .ingest import load_file
from .vector_store import rebuild_documents_to_vector_store


def rebuild_vector_store(db: Session):
    documents = db.query(Document).all()

    all_chunks: List[LCDocument] = []

    valid_document_count = 0

    for doc in documents:
        file_path = Path(doc.file_path)

        if not file_path.exists():
            continue

        valid_document_count += 1

        chunks = load_file(str(file_path))

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "file_path": doc.file_path,
                }
            )

        all_chunks.extend(chunks)

    rebuild_documents_to_vector_store(all_chunks)

    return {
        "message": "向量库重建成功",
        "document_count": valid_document_count,
        "chunk_count": len(all_chunks),
    }