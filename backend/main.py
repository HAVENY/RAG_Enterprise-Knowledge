from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Base, engine, get_db
from .ingest import ingest_file
from .models import Document
from .rag import rag_answer
from .schemas import QuestionRequest


settings = get_settings()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="企业内部知识库系统")

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

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = settings.uploads_dir / file.filename

    content_bytes = await file.read()
    file_path.write_bytes(content_bytes)

    document = Document(
        filename=file.filename,
        file_path=str(file_path),
        content="",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        content_text = ingest_file(
            file_path=str(file_path),
            document_id=document.id,
            db=db,
        )

        document.content = content_text
        db.commit()
        db.refresh(document)

    except Exception as e:
        db.delete(document)
        db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"文档解析或入库失败：{str(e)}",
        )

    return {
        "message": "文件上传并解析成功",
        "document_id": document.id,
        "filename": document.filename,
        "file_path": document.file_path,
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
        }
        for doc in documents
    ]


@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    result = rag_answer(
        question=request.question,
        db=db,
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }