import json
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx", "md", "csv", "xlsx"]


st.set_page_config(
    page_title="企业知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 2.2rem;
            padding-bottom: 6rem;
        }

        section[data-testid="stSidebar"] {
            width: 340px !important;
        }

        .app-title {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: 0.25rem;
            margin-bottom: 1.1rem;
        }

        .status-line {
            color: #6b7280;
            font-size: 0.86rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #eeeeee;
            margin-bottom: 1rem;
        }

        .source-card {
            border: 1px solid #e7e7e7;
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.6rem;
            background: #fafafa;
        }

        .source-meta {
            color: #6b7280;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "qa_history": [],
        "last_sources": [],
        "selected_doc_id": None,
        "selected_doc_name": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api_get(path: str, timeout: int = 30) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict | None = None, timeout: int = 120) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_documents() -> list[dict]:
    return api_get("/documents", timeout=30)


def fetch_chunks(document_id: int) -> dict:
    return api_get(f"/documents/{document_id}/chunks", timeout=30)


def fetch_history(limit: int = 20) -> list[dict]:
    return api_get(f"/history?limit={limit}", timeout=30)


def upload_document(file) -> dict:
    files = {
        "file": (
            file.name,
            file.getvalue(),
            file.type or "application/octet-stream",
        )
    }
    response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def delete_document(document_id: int) -> dict:
    response = requests.delete(f"{API_BASE_URL}/documents/{document_id}", timeout=120)
    response.raise_for_status()
    return response.json()


def rebuild_index() -> dict:
    return api_post("/documents/rebuild", timeout=300)


def ask_question(
    question: str,
    provider: str,
    model_level: str,
    allow_general_answer: bool,
) -> dict:
    return api_post(
        "/ask",
        payload={
            "question": question,
            "provider": provider,
            "model_level": model_level,
            "allow_general_answer": allow_general_answer,
        },
        timeout=120,
    )


def normalize_sources(raw_sources: Any) -> list[dict]:
    if isinstance(raw_sources, list):
        return raw_sources

    if isinstance(raw_sources, str) and raw_sources.strip():
        try:
            parsed = json.loads(raw_sources)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    return []


def render_topk(sources: list[dict]) -> None:
    if not sources:
        st.info("本次回答没有返回知识库来源。")
        return

    rows = []
    for index, source in enumerate(sources, start=1):
        rank = source.get("rank") or index
        score = source.get("score")
        if score is None:
            score = max(0, 1 - (index - 1) * 0.1)

        rows.append(
            {
                "TopK": f"Top {rank}",
                "score": round(float(score), 4),
                "raw_score": source.get("raw_score"),
                "document_id": source.get("document_id"),
                "chunk_index": source.get("chunk_index"),
                "source": source.get("source"),
                "preview": source.get("content_preview", ""),
            }
        )

    df = pd.DataFrame(rows)

    left, right = st.columns([0.9, 1.1])
    with left:
        st.bar_chart(df.set_index("TopK")[["score"]])

    with right:
        st.dataframe(
            df[["TopK", "score", "raw_score", "document_id", "chunk_index"]],
            use_container_width=True,
            hide_index=True,
        )

def render_message(item: dict) -> None:
    with st.chat_message("user"):
        st.write(item.get("question", ""))

    with st.chat_message("assistant"):
        st.write(item.get("answer", ""))
        st.caption(
            f"供应商：{item.get('provider', '未知')} | "
            f"档位：{item.get('model_level', '未知')} | "
            f"模型：{item.get('model', '未知')} | "
            f"模式：{item.get('mode', '未知')}"
        )

        sources = normalize_sources(item.get("sources", []))
        if sources:
            with st.expander(f"来源与 Top-K 检索，共 {len(sources)} 条", expanded=False):
                render_topk(sources)


def render_chunk_viewer(document_id: int) -> None:
    try:
        data = fetch_chunks(document_id)
    except Exception as exc:
        st.error(f"获取切片失败：{exc}")
        return

    chunks = data.get("chunks", [])
    st.caption(f"切片数量：{data.get('chunk_count', len(chunks))}")

    keyword = st.text_input(
        "筛选切片",
        placeholder="输入关键词",
        key=f"chunk_filter_{document_id}",
    )

    if keyword.strip():
        chunks = [
            chunk
            for chunk in chunks
            if keyword.lower() in chunk.get("chunk_text", "").lower()
        ]
        st.caption(f"匹配切片：{len(chunks)}")

    for chunk in chunks:
        chunk_index = chunk.get("chunk_index")
        chunk_text = chunk.get("chunk_text", "")
        with st.expander(f"Chunk {chunk_index} · {len(chunk_text)} 字", expanded=False):
            st.text_area(
                "切片内容",
                value=chunk_text,
                height=160,
                key=f"chunk_text_{document_id}_{chunk_index}",
                label_visibility="collapsed",
            )


def render_document_panel() -> None:
    st.markdown("### 文档")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("上传并入库", use_container_width=True):
        success_count = 0
        progress = st.progress(0)
        for index, file in enumerate(uploaded_files, start=1):
            try:
                upload_document(file)
                success_count += 1
            except Exception as exc:
                st.error(f"{file.name} 上传失败：{exc}")
            progress.progress(index / len(uploaded_files))
        st.success(f"完成上传：{success_count}/{len(uploaded_files)}")

    st.divider()

    try:
        documents = fetch_documents()
    except Exception as exc:
        st.warning(f"无法获取文档列表：{exc}")
        return

    if not documents:
        st.info("暂无文档。")
        return

    for doc in documents:
        doc_id = doc.get("id")
        filename = doc.get("filename", "未命名文档")
        chunk_count = doc.get("chunk_count", 0)

        with st.expander(f"{filename} · {chunk_count} chunks", expanded=False):
            st.caption(f"ID: {doc_id}")
            if doc.get("created_at"):
                st.caption(f"上传时间：{doc.get('created_at')}")

            col_view, col_delete = st.columns(2)
            with col_view:
                if st.button("查看切片", key=f"view_{doc_id}", use_container_width=True):
                    st.session_state.selected_doc_id = doc_id
                    st.session_state.selected_doc_name = filename
            with col_delete:
                if st.button("删除", key=f"delete_{doc_id}", use_container_width=True):
                    try:
                        delete_document(doc_id)
                        st.success("已删除并同步索引。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

            if st.session_state.selected_doc_id == doc_id:
                render_chunk_viewer(doc_id)


def render_knowledge_actions() -> None:
    st.markdown("### 知识库")

    col_refresh, col_clear = st.columns(2)
    with col_refresh:
        if st.button("刷新", use_container_width=True):
            st.rerun()
    with col_clear:
        if st.button("清空对话", use_container_width=True):
            st.session_state.qa_history = []
            st.session_state.last_sources = []
            st.rerun()

    if st.button("重建索引", use_container_width=True):
        try:
            with st.spinner("正在重建索引..."):
                result = rebuild_index()
            st.success(result.get("message", "重建完成"))
            st.caption(
                f"文档 {result.get('document_count', 0)} · "
                f"切片 {result.get('chunk_count', 0)} · "
                f"跳过 {result.get('skipped_count', 0)}"
            )
        except Exception as exc:
            st.error(f"重建失败：{exc}")


def render_history_panel() -> None:
    st.markdown("### 历史")
    try:
        histories = fetch_history(limit=10)
    except Exception as exc:
        st.caption(f"暂时无法读取历史：{exc}")
        return

    if not histories:
        st.caption("暂无历史记录。")
        return

    for item in histories:
        with st.expander(item.get("question", "未命名问题")[:40], expanded=False):
            st.write(item.get("answer", ""))
            st.caption(item.get("created_at", ""))
            sources = normalize_sources(item.get("sources"))
            if sources:
                st.caption(f"来源：{len(sources)} 条")


def render_sidebar() -> tuple[str, str, bool]:
    with st.sidebar:
        st.markdown("## 企业知识库")
        st.caption("RAG 文档问答")

        st.divider()

        provider_label = st.selectbox("模型供应商", ["Qwen", "MIMO"])
        provider = {"Qwen": "qwen", "MIMO": "mimo"}[provider_label]

        model_label = st.segmented_control(
            "模型档位",
            ["快速", "标准", "强力"],
            default="标准",
        )
        model_level = {"快速": "fast", "标准": "default", "强力": "strong"}[model_label]

        allow_general_answer = st.toggle(
            "无检索结果时允许通用回答",
            value=False,
        )

        st.divider()
        render_knowledge_actions()

        st.divider()
        render_document_panel()

        st.divider()
        render_history_panel()

    return provider, model_level, allow_general_answer


def render_header(provider: str, model_level: str, allow_general_answer: bool) -> None:
    st.markdown('<h1 class="app-title">企业知识问答系统</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">基于企业文档的检索增强问答，支持来源追溯与 Top-K 可视化。</div>',
        unsafe_allow_html=True,
    )
    mode = "允许通用回答" if allow_general_answer else "仅基于知识库回答"
    st.markdown(
        f'<div class="status-line">模型：{provider} / {model_level} · {mode}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    if st.session_state.qa_history:
        return

    st.markdown("#### 可以这样开始")
    col_summary, col_lookup, col_compare = st.columns(3)
    with col_summary:
        st.container(border=True).markdown("**总结文档**  \n总结某份文档的主要内容。")
    with col_lookup:
        st.container(border=True).markdown("**查询规则**  \n某个流程需要哪些材料？")
    with col_compare:
        st.container(border=True).markdown("**对比分析**  \n对比不同文档中的相关描述。")


def handle_question(provider: str, model_level: str, allow_general_answer: bool) -> None:
    question = st.chat_input("向企业知识库提问...")
    if not question:
        return

    if not question.strip():
        st.warning("请先输入问题。")
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                result = ask_question(
                    question=question,
                    provider=provider,
                    model_level=model_level,
                    allow_general_answer=allow_general_answer,
                )
            except Exception as exc:
                st.error(f"问答失败：{exc}")
                return

        answer = result.get("answer", "")
        sources = normalize_sources(result.get("sources", []))

        st.write(answer)
        st.caption(
            f"供应商：{result.get('provider', provider)} | "
            f"档位：{result.get('model_level', model_level)} | "
            f"模型：{result.get('model', '未知')} | "
            f"模式：{result.get('mode', '未知')}"
        )

        if sources:
            with st.expander(f"来源与 Top-K 检索，共 {len(sources)} 条", expanded=True):
                render_topk(sources)

    st.session_state.qa_history.append(
        {
            "question": question,
            "answer": answer,
            "provider": result.get("provider", provider),
            "model_level": result.get("model_level", model_level),
            "model": result.get("model"),
            "mode": result.get("mode"),
            "sources": sources,
        }
    )
    st.session_state.last_sources = sources
    st.rerun()


def main() -> None:
    inject_styles()
    init_state()

    provider, model_level, allow_general_answer = render_sidebar()
    render_header(provider, model_level, allow_general_answer)

    if st.session_state.last_sources:
        with st.expander(
            f"最近一次检索 Top-K，共 {len(st.session_state.last_sources)} 条",
            expanded=False,
        ):
            render_topk(st.session_state.last_sources)

    render_empty_state()

    for item in st.session_state.qa_history:
        render_message(item)

    handle_question(provider, model_level, allow_general_answer)


main()
