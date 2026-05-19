# Enterprise Knowledge Base

企业知识库问答系统，基于 **FastAPI + Streamlit + LangChain + FAISS + MySQL** 构建。

项目目标是把企业内部文档解析成可检索的知识片段，并通过 RAG（Retrieval-Augmented Generation）方式调用大模型生成带来源依据的回答。

---

## 1. 项目架构

```text
[Streamlit Frontend]
        |
        v
[FastAPI Backend]
   |
   +-- Document API
   |      +-- 文件上传
   |      +-- 文档列表
   |      +-- 文档切片查看
   |      +-- 文档删除
   |
   +-- Ingest Pipeline
   |      +-- 文档解析
   |      +-- 文本切片
   |      +-- 元数据入库
   |      +-- 向量索引写入
   |
   +-- RAG Pipeline
   |      +-- 问题向量检索
   |      +-- Top-K 片段召回
   |      +-- Prompt 组装
   |      +-- 大模型回答
   |      +-- 来源片段返回
   |
   +-- History API
          +-- 问答历史入库
          +-- 历史记录查询
          +-- 历史记录删除
```

### 1.1 后端

后端使用 FastAPI 提供 API 服务，主要负责：

- 接收和保存上传文档
- 解析文档内容并生成切片
- 写入 MySQL 元数据
- 写入和重建 FAISS 向量索引
- 执行 RAG 检索问答
- 返回答案、来源片段和 Top-K 分数
- 保存问答历史

### 1.2 前端

前端使用 Streamlit 构建，当前已重构为更接近 ChatGPT 的单页聊天体验：

- 主区域聚焦问答对话
- 左侧栏集中放置模型设置、知识库操作、文档管理和历史记录
- 来源追溯和 Top-K 可视化放在回答下方的展开区
- 最近一次检索结果可在主区快速查看

### 1.3 数据层

数据层由 MySQL 和 FAISS 组成：

- MySQL 保存文档记录、切片记录和问答历史。
- FAISS 保存文本切片向量索引，用于相似度检索。
- 上传文件保存在本地文件目录中，数据库保存文件元数据和解析后的文本信息。

---

## 2. 构建思路

### 2.1 文档入库

文档入库流程：

1. 用户在前端上传文档。
2. 后端保存原始文件。
3. 根据文件类型提取正文内容。
4. 将正文拆分成 chunk。
5. 将文档信息和 chunk 信息写入 MySQL。
6. 将 chunk 写入 FAISS 向量索引。

当前支持的文件格式：

- `txt`
- `md`
- `pdf`
- `docx`
- `csv`
- `xlsx`

### 2.2 知识库问答

问答流程：

1. 用户输入问题。
2. 后端加载 FAISS 索引。
3. 根据问题召回最相关的 Top-K 文档片段。
4. 将召回片段和用户问题组装成 RAG Prompt。
5. 调用指定大模型生成回答。
6. 返回回答内容、模型信息、来源片段和检索分数。
7. 将问答结果写入历史记录。

### 2.3 索引重建

重建流程用于保证数据库切片和 FAISS 索引保持一致：

1. 清理旧的向量索引。
2. 清理旧的切片记录。
3. 重新遍历已有文档。
4. 重新解析、切片、入库。
5. 重新生成 FAISS 索引。

删除文档后也会触发索引同步，避免已删除文档继续参与检索。

### 2.4 模型调用

系统通过统一的 LLM 调用层管理不同模型供应商。

当前支持：

- Qwen
- MIMO

模型档位：

- `fast`
- `default`
- `strong`

### 2.5 Embedding

系统优先使用本地 HuggingFace embedding 模型生成向量。

如果本地模型不可用，会降级到本地哈希 embedding，保证开发环境下基础流程仍可运行。生产环境建议使用稳定的中文或多语言 embedding 模型。

---

## 3. 目前功能

### 3.1 文档管理

- 批量上传文档
- 查看文档列表
- 查看每个文档的切片数量
- 查看文档切片全文
- 按关键词筛选当前文档切片
- 删除文档并同步更新索引

### 3.2 知识库索引

- 上传后自动写入向量索引
- 支持手动重建知识库索引
- 删除文档后同步更新索引
- 支持索引为空时的提示和兜底逻辑

### 3.3 问答能力

- 基于知识库片段回答问题
- 返回来源片段，便于追溯
- 返回 Top-K 检索排名和相似度分数
- 支持模型供应商切换
- 支持模型档位切换
- 支持在无知识库结果时选择是否允许通用回答

### 3.4 问答历史

- `/ask` 回答后自动写入问答历史
- 支持读取最近问答历史
- 支持删除指定历史记录
- 前端侧边栏展示最近历史记录摘要

### 3.5 前端体验

- 主区采用聊天式布局
- 侧边栏承载知识库管理能力
- 来源片段以展开区展示
- Top-K 检索结果支持柱状图和表格
- 代码结构拆分为 API 调用、数据处理、组件渲染和主流程

---

## 4. 主要 API

```http
GET /
POST /upload
GET /documents
GET /documents/{document_id}/chunks
DELETE /documents/{document_id}
POST /documents/rebuild
POST /ask
POST /chat
GET /history
DELETE /history/{history_id}
```

---

## 5. 本地运行

### 5.1 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.2 配置环境变量

复制 `.env_example` 为 `.env`，然后填写数据库连接和模型 API Key。

关键配置：

- `DATABASE_URL`
- `EMBEDDING_MODEL_NAME`
- `DEFAULT_LLM_PROVIDER`
- `DEFAULT_MODEL_LEVEL`
- `DASHSCOPE_API_KEY`
- `MIMO_API_KEY`

### 5.3 启动后端

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.4 启动前端

```powershell
streamlit run frontend/app.py
```

---

## 6. 仓库说明

以下内容不应提交到 Git：

- `.env`
- 本地虚拟环境
- Python 缓存文件
- 上传的业务文档
- FAISS 本地索引文件
- 本地数据库文件

仓库中只保留源码、依赖说明、配置模板和项目文档。
