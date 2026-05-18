import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="企业知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 页面样式：ChatGPT 风格
# =========================

st.markdown(
    """
    <style>
    .block-container {
    padding-top: 5.5rem;
    padding-bottom: 1rem;
    max-width: 1100px;
    }

    section[data-testid="stSidebar"] {
        width: 320px !important;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #666;
        font-size: 14px;
        margin-bottom: 1.2rem;
    }

    .chat-header {
        border-bottom: 1px solid #eee;
        padding-bottom: 0.8rem;
        margin-bottom: 1rem;
    }

    .source-card {
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        background-color: #fafafa;
    }

    .small-caption {
        font-size: 12px;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Session State
# =========================

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# =========================
# 工具函数
# =========================

def get_documents():
    try:
        response = requests.get(
            f"{API_BASE_URL}/documents",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"暂时无法获取文档列表：{e}")
        return None


def upload_document(file):
    try:
        files = {
            "file": (
                file.name,
                file.getvalue(),
                file.type or "text/plain",
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"上传失败：{e}")

def fetch_documents():
    response = requests.get(
        f"{API_BASE_URL}/documents",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_chunks(document_id: int):
    response = requests.get(
        f"{API_BASE_URL}/documents/{document_id}/chunks",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def rebuild_vector_store():
    try:
        response = requests.post(
            f"{API_BASE_URL}/rebuild",
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"重建知识库失败：{e}")
        return None


def delete_document(document_id):
    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{document_id}",
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"删除失败：{e}")
        return None


def ask_question(question, provider, model_level, allow_general_answer):
    try:
        payload = {
            "question": question,
            "provider": provider,
            "model_level": model_level,
            "allow_general_answer": allow_general_answer,
        }

        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"问答失败：{e}")
        return None


# =========================
# 左侧边栏
# =========================

with st.sidebar:
    st.markdown("## 📚 企业知识库")
    st.caption("RAG 文档问答系统")

    st.divider()

    # ---------- 模型设置 ----------
    st.markdown("### 🤖 模型设置")

    provider_label = st.selectbox(
        "模型供应商",
        options=["Qwen", "MIMO"],
        index=0,
    )

    provider_map = {
        "Qwen": "qwen",
        "MIMO": "mimo",
    }
    provider = provider_map[provider_label]

    model_level_label = st.selectbox(
        "模型模式",
        options=["快速模式", "标准模式", "强力模式"],
        index=1,
    )

    model_level_map = {
        "快速模式": "fast",
        "标准模式": "default",
        "强力模式": "strong",
    }
    model_level = model_level_map[model_level_label]

    model_desc = {
        "fast": "响应更快，适合简单问答。",
        "default": "效果与成本均衡，适合日常知识库问答。",
        "strong": "适合复杂总结、长文本分析和严谨推理。",
    }

    st.caption(model_desc[model_level])

    allow_general_answer = st.checkbox(
        "知识库无结果时允许通用回答",
        value=False,
    )

    st.divider()

    # ---------- 文档上传 ----------
    uploaded_files = st.file_uploader(
    "上传文档",
    type=["txt", "pdf", "docx", "md", "csv", "xlsx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个文件")

        with st.expander("查看待上传文件", expanded=False):
            for file in uploaded_files:
                st.write(f"- {file.name}")

        if st.button("批量上传并入库", use_container_width=True):
            success_count = 0
            fail_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for index, file in enumerate(uploaded_files, start=1):
                status_text.write(f"正在上传：{file.name} ({index}/{len(uploaded_files)})")

                try:
                    result = upload_document(file)

                    if result:
                        success_count += 1
                    else:
                        fail_count += 1

                except Exception as e:
                    fail_count += 1
                    st.error(f"{file.name} 上传失败：{e}")

                progress_bar.progress(index / len(uploaded_files))

            if success_count > 0:
                st.success(f"批量上传完成：成功 {success_count} 个，失败 {fail_count} 个")

            if fail_count > 0:
                st.warning("部分文件上传失败，请查看上方错误信息。")
                
        st.divider()

    # ---------- 已上传文档 ----------
    st.markdown("### 📄 已上传文档")

    if st.button("刷新文档列表", use_container_width=True):
        st.session_state["refresh_documents"] = True

    try:
        documents = fetch_documents()

        if not documents:
            st.info("暂无已上传文档。")
        else:
            for doc in documents:
                doc_id = doc.get("id")
                filename = doc.get("filename", "")
                chunk_count = doc.get("chunk_count", 0)
                file_path = doc.get("file_path", "")
                file_exists = doc.get("file_exists", None)

                doc_title = f"{filename}｜ID: {doc_id}｜切片: {chunk_count}"

                if file_exists is False:
                    doc_title = f"⚠️ {doc_title}｜原文件丢失"

                with st.expander(doc_title, expanded=False):
                    st.caption(f"文件路径：{file_path}")

                    created_at = doc.get("created_at")
                    if created_at:
                        st.caption(f"上传时间：{created_at}")

                    view_btn = st.button(
                        "查看切片",
                        key=f"view_chunks_{doc_id}",
                        use_container_width=True,
                    )

                    if view_btn:
                        st.session_state["sidebar_selected_doc_id"] = doc_id
                        st.session_state["sidebar_selected_doc_name"] = filename

                    if st.session_state.get("sidebar_selected_doc_id") == doc_id:
                        try:
                            chunk_data = fetch_chunks(doc_id)
                            chunks = chunk_data.get("chunks", [])

                            st.write(f"切片数量：{chunk_data.get('chunk_count', 0)}")

                            if not chunks:
                                st.warning("该文档暂无切片。")
                            else:
                                keyword = st.text_input(
                                    "搜索当前文档切片",
                                    placeholder="输入关键词筛选 chunk",
                                    key=f"sidebar_chunk_keyword_{doc_id}",
                                )

                                filtered_chunks = chunks

                                if keyword.strip():
                                    filtered_chunks = [
                                        chunk
                                        for chunk in chunks
                                        if keyword.lower()
                                        in chunk.get("chunk_text", "").lower()
                                    ]

                                    st.info(f"匹配到 {len(filtered_chunks)} 个切片")

                                for chunk in filtered_chunks:
                                    chunk_index = chunk.get("chunk_index")
                                    chunk_text = chunk.get("chunk_text", "")
                                    vector_id = chunk.get("vector_id")

                                    with st.expander(
                                        f"Chunk #{chunk_index}｜长度：{len(chunk_text)}",
                                        expanded=False,
                                    ):
                                        st.caption(f"chunk_index: {chunk_index}")
                                        st.caption(f"vector_id: {vector_id}")

                                        st.text_area(
                                            "chunk_text",
                                            value=chunk_text,
                                            height=180,
                                            key=f"sidebar_chunk_text_{doc_id}_{chunk_index}",
                                        )

                        except Exception as e:
                            st.error(f"获取切片失败：{e}")

    except Exception as e:
        st.error(f"获取文档列表失败：{e}")
                
                
                
                
    # ---------- 系统操作 ----------
    st.markdown("### ⚙️ 知识库操作")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("刷新", use_container_width=True):
            st.rerun()

    with col_b:
        if st.button("清空对话", use_container_width=True):
            st.session_state.qa_history = []
            st.session_state.last_sources = []
            st.rerun()

    if st.button("重建知识库索引", use_container_width=True):
        try:
            with st.spinner("正在重建 FAISS 索引和文档切片..."):
                response = requests.post(
                    f"{API_BASE_URL}/rebuild",
                    timeout=180,
                )

            if response.ok:
                rebuild_result = response.json()

                st.success(rebuild_result.get("message", "重建成功"))
                st.write(f"文档数量：{rebuild_result.get('document_count', 0)}")
                st.write(f"切片数量：{rebuild_result.get('chunk_count', 0)}")

                with st.expander("查看重建详情"):
                    st.json(rebuild_result)
            else:
                st.error("重建失败")
                st.code(response.text)

        except Exception as e:
            st.error(f"重建请求异常：{e}")

    st.divider()

    

# =========================
# 主区域：ChatGPT 风格问答区
# =========================

st.markdown(
    """
    <div class="chat-header">
        <div class="main-title">企业知识问答系统</div>
        <div class="sub-title">基于上传文档进行检索增强问答，支持多模型切换与来源追踪。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 当前配置提示
st.info(
    f"当前配置：{provider_label} / {model_level_label} / "
    f"{'允许通用回答' if allow_general_answer else '仅基于知识库回答'}"
)

# 没有历史时展示引导卡片
if not st.session_state.qa_history:
    st.markdown("### 你可以这样提问")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.container(border=True).markdown(
            "**总结文档**  \n请总结当前知识库中某份文档的主要内容。"
        )

    with col2:
        st.container(border=True).markdown(
            "**查询流程**  \n公司报销流程是什么？需要哪些材料？"
        )

    with col3:
        st.container(border=True).markdown(
            "**对比分析**  \n请对比不同文档中关于系统开发流程的描述。"
        )

    st.divider()


# 展示历史对话
for item in st.session_state.qa_history:
    with st.chat_message("user"):
        st.write(item.get("question", ""))

    with st.chat_message("assistant"):
        st.write(item.get("answer", ""))

        st.caption(
            f"模型供应商：{item.get('provider', provider)} | "
            f"模型模式：{item.get('model_level', model_level)} | "
            f"实际模型：{item.get('model', '未知')} | "
            f"回答模式：{item.get('mode', '未知')}"
        )

        sources = item.get("sources", [])

        if sources:
            with st.expander(f"查看参考来源，共 {len(sources)} 条"):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**来源 {i}**")
                    st.write(f"文档 ID：{source.get('document_id')}")
                    st.write(f"切片序号：{source.get('chunk_index')}")
                    st.write(f"文件路径：{source.get('source')}")
                    st.caption(source.get("content_preview", ""))


# 底部输入框
question = st.chat_input("向企业知识库提问...")

if question:
    if not question.strip():
        st.warning("请先输入问题。")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并调用大模型生成回答..."):
                result = ask_question(
                    question=question,
                    provider=provider,
                    model_level=model_level,
                    allow_general_answer=allow_general_answer,
                )

            if result:
                answer = result.get("answer", "")
                sources = result.get("sources", [])

                st.write(answer)

                st.caption(
                    f"模型供应商：{result.get('provider', provider)} | "
                    f"模型模式：{result.get('model_level', model_level)} | "
                    f"实际模型：{result.get('model', '未知')} | "
                    f"回答模式：{result.get('mode', '未知')}"
                )

                if sources:
                    with st.expander(f"查看参考来源，共 {len(sources)} 条"):
                        for i, source in enumerate(sources, start=1):
                            st.markdown(f"**来源 {i}**")
                            st.write(f"文档 ID：{source.get('document_id')}")
                            st.write(f"切片序号：{source.get('chunk_index')}")
                            st.write(f"文件路径：{source.get('source')}")
                            st.caption(source.get("content_preview", ""))
                else:
                    st.info("本次回答没有返回知识库来源。")

                st.session_state.qa_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "provider": result.get("provider", provider),
                        "model_level": result.get("model_level", model_level),
                        "model": result.get("model"),
                        "mode": result.get("mode"),
                        "sources": sources,
                        "raw_result": result,
                    }
                )

                st.session_state.last_sources = sources
                st.rerun()