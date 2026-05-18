from sqlalchemy.orm import Session

from .ingest import ingest_file
from .models import Document, DocumentChunk
from .storage import resolve_document_path
from .vector_store import reset_faiss_index


def rebuild_vector_store(db: Session):
    documents = db.query(Document).order_by(Document.id.asc()).all()

    reset_faiss_index()
    db.query(DocumentChunk).delete()
    db.commit()

    rebuilt_documents = []
    skipped_documents = []
    total_chunk_count = 0

    for document in documents:
        file_path = resolve_document_path(document.file_path, document.filename)

        if not file_path.exists():
            skipped_documents.append(
                {
                    "document_id": document.id,
                    "filename": document.filename,
                    "old_file_path": document.file_path,
                    "reason": "file_not_found",
                }
            )
            continue

        canonical_path = str(file_path)
        if document.file_path != canonical_path:
            document.file_path = canonical_path
            db.commit()

        try:
            ingest_result = ingest_file(
                file_path=canonical_path,
                document_id=document.id,
                db=db,
            )
        except Exception as e:
            db.rollback()
            skipped_documents.append(
                {
                    "document_id": document.id,
                    "filename": document.filename,
                    "file_path": canonical_path,
                    "reason": str(e),
                }
            )
            continue

        document.content = ingest_result["text"]
        db.commit()

        total_chunk_count += ingest_result["chunk_count"]
        rebuilt_documents.append(
            {
                "document_id": document.id,
                "filename": document.filename,
                "file_path": document.file_path,
                "chunk_count": ingest_result["chunk_count"],
                "text_length": ingest_result["text_length"],
            }
        )

    return {
        "message": "索引重建完成",
        "document_count": len(documents),
        "rebuilt_count": len(rebuilt_documents),
        "skipped_count": len(skipped_documents),
        "chunk_count": total_chunk_count,
        "documents": rebuilt_documents,
        "skipped_documents": skipped_documents,
    }
