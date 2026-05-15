# Enterprise Knowledge Base

一个基于 **FastAPI + Streamlit + LangChain + FAISS** 的企业知识库问答系统。  
核心目标：让内部文档可被结构化检索，并通过大模型进行可追溯问答。

---

## 1. 架构说明

### 1.1 总体架构

```text
[Frontend / Streamlit]
        |
        v
[Backend / FastAPI API Layer]
   |            |
   |            +--> [LLM Router: Qwen / MIMO]
   |
   +--> [Ingest Pipeline]
           |
           +--> 文档解析（txt/md/pdf/docx/csv/xlsx）
           +--> 文本切片（chunk）
           +--> 向量化（Embedding）
           +--> FAISS 向量索引
           +--> SQLite 元数据（文档/切片）

问答路径：
用户问题 -> 向量检索 TopK -> 组装上下文 Prompt -> LLM 流式生成 -> 前端展示
```

### 1.2 模块划分

- `backend/main.py`：API 入口，提供上传、问答、文档列表等接口。
- `backend/ingest.py`：文档解析与切片入库流程。
- `backend/vector_store.py`：FAISS 索引读写与增量更新。
- `backend/rag.py`：RAG 检索与回答编排逻辑。
- `backend/llm.py`：多模型统一调用层（Provider + Model Level）。
- `backend/models.py`：SQLite 数据模型（`documents`、`document_chunks`）。
- `frontend/app.py`：Streamlit 交互界面。

### 1.3 数据流（上传到问答）

1. 上传文档到后端。
2. 解析文档正文并切片。
3. 切片写入 SQLite，同时写入/追加 FAISS 索引。
4. 用户提问后进行相似度检索（TopK）。
5. 拼接检索片段进入 Prompt，调用模型生成答案。
6. 返回答案 + 来源片段，实现可追溯问答。

---

## 2. 关键 Prompt 与 Vibe 思路

### 2.1 Prompt 设计目标

- **优先依据知识库回答**：避免脱离企业文档“自由发挥”。
- **不确定时显式说明**：信息不足时明确“无法完整确定”，降低幻觉风险。
- **输出结构清晰**：对流程/制度类问题优先分点表达，便于业务同学阅读。

### 2.2 关键 Prompt 结构（RAG）

RAG Prompt 核心包含三段：

1. **角色与约束**：企业知识库问答助手，禁止编造内部信息。  
2. **检索上下文**：注入 TopK 片段作为唯一高优先级依据。  
3. **用户问题与输出要求**：围绕问题作答，信息不足时明确说明。

可在 `backend/rag.py` 的 `build_rag_prompt` 中继续优化：
- 增加答案模板（结论 / 依据 / 风险 / 下一步）
- 增加引用标记规范（如片段编号）
- 区分事实问题与总结问题的不同语气与粒度

### 2.3 Vibe（回答风格）思路

当前策略偏向：
- **专业稳健**：尽量事实化与中性表达。
- **可审计**：返回来源元数据，便于复核。
- **可运营**：支持模型档位切换（fast/default/strong），平衡速度与效果。

---

## 3. AI 调用逻辑（流式）

### 3.1 目标

将问答接口升级为 **Streaming**：后端边生成边返回，前端逐段展示，降低首字延迟并提升交互体验。

### 3.2 推荐实现方式

- 后端：FastAPI 使用 `StreamingResponse`。
- 传输：建议 `text/event-stream`（SSE）或 chunked 文本流。
- 前端：Streamlit 使用流式读取并实时刷新回答区。

### 3.3 后端关键流程（逻辑）

1. 完成检索并构造 Prompt。
2. 选择 Provider 与 Model。
3. 调用 LLM 的流式接口（`stream=True`）。
4. 将增量 token 按事件持续 yield 给客户端。
5. 结束时返回完成标记与来源信息。

### 3.4 前端关键流程（逻辑）

1. 发起问答请求到流式接口（如 `/ask/stream`）。
2. 持续读取事件流。
3. 每次收到 token 即刷新页面文本。
4. 流结束后展示来源片段与模型信息。

---

## 4. 部署步骤

### 4.1 环境准备

- Python 3.10+（推荐 3.11/3.12）
- 可联网下载依赖与模型
- Windows / Linux / macOS 均可

### 4.2 安装依赖

```powershell
# 进入项目根目录
cd Enterprise_knowledge_base

# 创建虚拟环境
python -m venv .venv

# 激活（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 4.3 配置环境变量

在项目根目录创建 `.env`（示例）：

```env
APP_ENV=development

# LLM 默认配置
DEFAULT_LLM_PROVIDER=qwen
DEFAULT_MODEL_LEVEL=default

# Qwen
DASHSCOPE_API_KEY=your_dashscope_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# MIMO
MIMO_API_KEY=your_mimo_key
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
```

### 4.4 启动后端

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4.5 启动前端

```powershell
streamlit run frontend/app.py
```

默认访问：
- 后端 API: `http://127.0.0.1:8000`
- 前端页面: `http://localhost:8501`

### 4.6 生产部署建议

- 使用 `gunicorn + uvicorn workers`（Linux）或容器化部署。
- 将 `.env` 中密钥迁移到安全配置中心。
- 为上传目录、数据库和向量索引配置持久化卷。
- 在网关层增加鉴权、限流与审计日志。

---

## 项目状态

当前已具备可运行的企业知识库原型：
- 多格式文档入库
- 向量检索 + RAG 问答
- 多模型切换
- 来源可追溯

下一步建议优先完成：
1. `/ask/stream` 流式问答接口落地
2. 检索质量评测与重排
3. 权限与多租户隔离
