from pathlib import Path

import fitz
import pandas as pd
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from .vector_store import save_chunks_to_faiss
from .models import DocumentChunk


def read_txt_or_md(file_path: str) -> str:
    path = Path(file_path)

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def read_pdf(file_path: str) -> str:
    text_list = []

    pdf = fitz.open(file_path)

    for page in pdf:
        text = page.get_text()
        if text:
            text_list.append(text)

    pdf.close()

    return "\n".join(text_list)


def read_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)

    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def read_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_string(index=False)


def read_xlsx(file_path: str) -> str:
    excel_file = pd.ExcelFile(file_path)

    sheets_text = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        sheet_text = f"【Sheet: {sheet_name}】\n{df.to_string(index=False)}"
        sheets_text.append(sheet_text)

    return "\n\n".join(sheets_text)


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix in [".txt", ".md"]:
        return read_txt_or_md(file_path)

    if suffix == ".pdf":
        return read_pdf(file_path)

    if suffix == ".docx":
        return read_docx(file_path)

    if suffix == ".csv":
        return read_csv(file_path)

    if suffix == ".xlsx":
        return read_xlsx(file_path)

    raise ValueError(f"暂不支持该文件格式：{suffix}")


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    return splitter.split_text(text)


def ingest_file(file_path: str, document_id: int, db: Session) -> str:
    """
    解析文档正文，切分为 chunk，写入 document_chunks 表，并写入 FAISS。
    """

    text = extract_text(file_path)

    if not text.strip():
        raise ValueError("文档内容为空，可能是扫描版 PDF 或暂不支持的格式。")

    chunks = split_text(text)

    metadatas = []

    for index, chunk in enumerate(chunks):
        db_chunk = DocumentChunk(
            document_id=document_id,
            chunk_text=chunk,
            chunk_index=index,
            vector_id=None,
        )

        db.add(db_chunk)

        metadatas.append(
            {
                "document_id": document_id,
                "chunk_index": index,
                "source": file_path,
            }
        )

    db.commit()

    save_chunks_to_faiss(
        chunks=chunks,
        metadatas=metadatas,
    )

    return text