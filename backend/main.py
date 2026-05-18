from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Base, engine, get_db
from .ingest import ingest_file
from .models import Document, DocumentChunk
from .rag import rag_answer
from .schemas import QuestionRequest
from .storage import get_upload_path, resolve_document_path, safe_upload_filename
from .vector_store import reset_faiss_index


settings = get_settings()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Enterprise Knowledge Base API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"message": "Enterprise Knowledge Base API is running"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    filename = safe_upload_filename(file.filename)
    existing_document = db.query(Document).filter(Document.filename == filename).first()
    if existing_document:
        raise HTTPException(status_code=409, detail="同名文件已存在，请先删除旧文件或重命名后再上传")

    file_path = get_upload_path(filename)
    file_path.write_bytes(await file.read())

    document = Document(
        filename=filename,
        file_path=str(file_path),
        content="",
    )

    db.add(document)

    try:
        db.commit()
        db.refresh(document)
    except IntegrityError:
        db.rollback()
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=409, detail="同名文件已存在，请先删除旧文件或重命名后再上传")

    try:
        ingest_result = ingest_file(
            file_path=str(file_path),
            document_id=document.id,
            db=db,
        )

        document.content = ingest_result["text"]
        db.commit()
        db.refresh(document)
    except Exception as e:
        db.delete(document)
        db.commit()
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"文档解析或入库失败：{str(e)}",
        )

    return {
        "message": "文件上传并解析成功",
        "document_id": document.id,
        "filename": document.filename,
        "file_path": document.file_path,
        "text_length": ingest_result["text_length"],
        "chunk_count": ingest_result["chunk_count"],
    }


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "created_at": doc.created_at,
            "chunk_count": db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc.id)
            .count(),
        }
        for doc in documents
    ]


@app.get("/documents/{document_id}/chunks")
def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    return {
        "document_id": document.id,
        "filename": document.filename,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "vector_id": chunk.vector_id,
            }
            for chunk in chunks
        ],
    }


@app.post("/rebuild")
def rebuild_index(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.id.asc()).all()

    if not documents:
        reset_faiss_index()
        return {
            "message": "暂无文档，无需重建索引",
            "document_count": 0,
            "rebuilt_count": 0,
            "skipped_count": 0,
            "chunk_count": 0,
            "documents": [],
            "skipped_documents": [],
        }

    try:
        reset_faiss_index()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清理 FAISS 索引失败：{str(e)}",
        )

    db.query(DocumentChunk).delete()
    db.commit()

    total_chunk_count = 0
    rebuilt_documents = []
    skipped_documents = []

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


@app.post("/documents/rebuild")
def rebuild_documents_index(db: Session = Depends(get_db)):
    return rebuild_index(db=db)


@app.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    file_path = resolve_document_path(document.file_path, document.filename)
    filename = document.filename

    db.delete(document)
    db.commit()

    if file_path.exists():
        file_path.unlink()

    rebuild_result = rebuild_index(db=db)

    return {
        "message": "文档删除成功，知识库索引已重建",
        "document_id": document_id,
        "filename": filename,
        "rebuild": rebuild_result,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    return rag_answer(
        question=request.question,
        db=db,
        provider=request.provider,
        model_level=request.model_level,
        allow_general_answer=request.allow_general_answer,
    )


@app.post("/chat")
def chat(request: QuestionRequest, db: Session = Depends(get_db)):
    return rag_answer(
        question=request.question,
        db=db,
        provider=request.provider,
        model_level=request.model_level,
        allow_general_answer=request.allow_general_answer,
    )
