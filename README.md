# Enterprise Knowledge Base

一个基于 **FastAPI + Streamlit + LangChain + FAISS + MySQL** 的企业知识库问答系统。

系统支持批量上传企业文档，解析后写入 MySQL 与 FAISS 向量索引，并通过 RAG 方式调用大模型生成可追溯回答。当前前端已经包含文档上传、文档管理、切片查看、知识库重建和知识问答能力。

---

## 1. 当前状态

当前已完成：

- 多格式文档上传：`txt`、`md`、`pdf`、`docx`、`csv`、`xlsx`
- 上传文件统一保存到 `data/uploads`
- 文档元数据与切片写入 MySQL
- FAISS 向量索引写入与加载
- 知识库索引重建
- 删除文档后自动重建索引
- 前端查看文档列表、切片数量和切片全文
- 前端支持按关键词筛选当前文档切片
- RAG 问答返回答案、来源片段、模型供应商、模型档位
- 支持 Qwen / MIMO 模型供应商切换
- embedding 优先使用本地 HuggingFace 模型，不可用时自动降级到本地哈希 embedding

当前验证结果：

```text
documents = 12
chunks = 87
missing_paths = []
faiss_index = index.faiss + index.pkl
```

---

## 2. 架构说明

```text
[Streamlit Frontend]
        |
        v
[FastAPI Backend]
   |
   +--> Upload / Document API
   |       |
   |       +--> data/uploads
   |       +--> MySQL: documents, document_chunks
   |
   +--> Ingest Pipeline
   |       |
   |       +--> 文档解析
   |       +--> 文本切片
   |       +--> embedding
   |       +--> FAISS
   |
   +--> RAG Pipeline
           |
           +--> FAISS TopK 检索
           +--> Prompt 组装
           +--> Qwen / MIMO
           +--> 答案 + 来源片段
```

### 2.1 主要模块

- `backend/main.py`：FastAPI 入口，提供上传、文档列表、切片查看、删除、重建、问答接口。
- `backend/config.py`：配置读取，默认读取项目根目录 `.env`。
- `backend/db.py`：SQLAlchemy 数据库连接。
- `backend/models.py`：MySQL 表模型，包含 `documents` 和 `document_chunks`。
- `backend/storage.py`：统一上传路径，兼容旧路径并修复历史混乱路径。
- `backend/ingest.py`：文档解析、切片、写入切片表、写入 FAISS。
- `backend/vector_store.py`：FAISS 读写、索引清理、embedding 获取与兜底。
- `backend/rag.py`：检索增强问答逻辑。
- `backend/llm.py`：Qwen / MIMO 统一调用。
- `frontend/app.py`：Streamlit 前端界面。

---

## 3. 路径规则

当前统一路径：

```text
data/uploads/
```

所有新上传文件都会保存到该目录。

历史上项目曾出现过旧路径，例如：

```text
uploads/
../uploads/
../data/uploads/
```

现在 `backend/storage.py` 会在重建索引时按文件名自动查找这些旧路径。如果找到文件，会把数据库中的 `file_path` 修正为当前统一路径；如果找不到，会把该文档写入 `skipped_documents`，不会让整个重建流程失败。

---

## 4. 数据库

当前运行配置使用 MySQL：

```env
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/enterprise_kb?charset=utf8mb4
```

代码里 `backend/config.py` 保留了 SQLite 默认值，只是兜底配置。实际运行以 `.env` 中的 `DATABASE_URL` 为准。

主要表：

- `documents`：文档记录，保存文件名、文件路径、解析文本、创建时间。
- `document_chunks`：文档切片，保存文档 ID、切片文本、切片序号。

---

## 5. 索引重建

后端提供两个等价接口：

```http
POST /rebuild
POST /documents/rebuild
```

重建流程：

1. 清空 `backend/vector_store/faiss_index` 中的索引文件。
2. 删除 MySQL 中的旧切片记录。
3. 遍历所有文档记录。
4. 使用 `resolve_document_path` 修正路径。
5. 重新解析文档并切片。
6. 写入 `document_chunks`。
7. 写入新的 FAISS 索引。
8. 返回成功文档、跳过文档和切片数量。

注意：Windows 下不再删除整个 `faiss_index` 目录，只清空目录内容，避免目录被占用时出现 `PermissionError`。

---

## 6. Embedding 策略

系统优先使用 `.env` 中的：

```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

加载方式为本地优先：

```python
model_kwargs={"device": "cpu", "local_files_only": True}
```

如果本地模型不可用，系统会自动降级到 `LocalHashEmbeddings`，保证上传、切片、重建和检索流程可以继续跑通。

生产环境建议准备稳定的中文或多语言 embedding 模型，并确保服务机器可以离线加载该模型。

---

## 7. 前端功能

Streamlit 前端当前包含：

- 批量上传文档
- 查看已上传文档
- 查看每个文档的切片数量
- 点击“查看切片”查看切片全文
- 按关键词筛选当前文档切片
- 删除文档
- 重建知识库索引
- 选择模型供应商：Qwen / MIMO
- 选择模型档位：fast / default / strong
- 知识库问答
- 查看回答来源片段

---

## 8. API 概览

### 健康检查

```http
GET /
```

### 上传文档

```http
POST /upload
```

### 文档列表

```http
GET /documents
```

### 查看文档切片

```http
GET /documents/{document_id}/chunks
```

### 删除文档

```http
DELETE /documents/{document_id}
```

删除后会自动重建索引。

### 重建索引

```http
POST /rebuild
POST /documents/rebuild
```

### 知识问答

```http
POST /ask
POST /chat
```

请求示例：

```json
{
  "question": "请总结这份文档的主要内容",
  "provider": "qwen",
  "model_level": "default",
  "allow_general_answer": false
}
```

---

## 9. 本地运行

### 9.1 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

当前项目本机也存在外层虚拟环境：

```powershell
..\EKB_env\Scripts\python.exe
```

### 9.2 配置环境变量

项目提供 `.env_example` 作为模板。真实 `.env` 不应提交到仓库。

需要重点配置：

- `DATABASE_URL`
- `DASHSCOPE_API_KEY`
- `MIMO_API_KEY`
- `EMBEDDING_MODEL_NAME`

### 9.3 启动后端

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

默认 API：

```text
http://127.0.0.1:8000
```

### 9.4 启动前端

```powershell
streamlit run frontend/app.py
```

默认页面：

```text
http://localhost:8501
```

---

## 10. Git 注意事项

`.gitignore` 已忽略：

- `.env`
- 虚拟环境
- `__pycache__`
- `data/uploads/*`
- `backend/vector_store/faiss_index/`
- 本地数据库文件

不要提交真实业务文件、真实 API Key、FAISS 索引文件或 Python 缓存文件。

---

## 11. 下一步建议

1. 清理已有 `__pycache__` 和历史根目录 `uploads` 脏文件。
2. 优化前端中文编码显示，确保所有文案稳定为 UTF-8。
3. 增加 `/ask/stream` 流式问答接口。
4. 增加检索评分、rerank 和命中片段高亮。
5. 增加权限控制、审计日志和多用户隔离。
