# Enterprise Knowledge Base

一个基于 FastAPI + Streamlit 的企业知识库项目骨架，包含基础的文档上传、入库和 RAG 问答流程。

## 项目结构

```text
enterprise_knowledge_base/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   ├── models.py
│   ├── rag.py
│   ├── ingest.py
│   ├── embeddings.py
│   ├── llm.py
│   └── vector_store/
│       └── faiss_index/
├── frontend/
│   └── app.py
├── data/
│   └── uploads/
├── requirements.txt
├── .env
└── README.md
```

## 1. 创建虚拟环境

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 3. 启动后端

```powershell
uvicorn backend.main:app --reload
```

## 4. 启动前端

```powershell
streamlit run frontend/app.py
```

## 当前能力

- 上传 `.txt` / `.md` 文档到 `data/uploads/`
- 将文档内容写入 SQLite
- 基于简单关键词匹配做检索
- 通过 `llm.py` 中的占位 Qwen 封装返回示例回答

## 后续建议

- 接入真实的 Embedding 模型
- 替换 `rag.py` 为 FAISS 向量检索
- 在 `llm.py` 中接入真实的 Qwen API
- 增加 PDF / DOCX 文档解析
