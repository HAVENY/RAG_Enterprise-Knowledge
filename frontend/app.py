import streamlit as st
import requests


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="企业知识问答系统", layout="wide")
st.title("企业知识问答系统")
st.caption("上传企业文档，并通过 RAG 问答快速检索内部知识。")


with st.sidebar:
    st.header("文档上传")

    uploaded_file = st.file_uploader(
        "请选择要上传的文档",
        type=["txt", "pdf", "docx", "md"],
    )

    if uploaded_file and st.button("上传并入库"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "text/plain",
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=60,
        )

        if response.ok:
            st.success(response.json()["message"])
        else:
            st.error(response.text)

    st.divider()

    st.header("模型设置")

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


st.subheader("知识问答")

question = st.text_area("输入你的问题")

if st.button("开始提问"):
    if not question.strip():
        st.warning("请先输入问题。")
    else:
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

        if response.ok:
            result = response.json()

            st.subheader("回答")
            st.write(result.get("answer", ""))

            st.divider()

            st.caption(f"模型供应商：{result.get('provider', provider)}")
            st.caption(f"模型模式：{result.get('model_level', model_level)}")

            if "model" in result:
                st.caption(f"实际调用模型：{result.get('model')}")

            if "mode" in result:
                st.caption(f"回答模式：{result.get('mode')}")

            sources = result.get("sources", [])

            if sources:
                st.subheader("参考来源")

                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**来源 {i}**")
                    st.write(f"文档 ID：{source.get('document_id')}")
                    st.write(f"切片序号：{source.get('chunk_index')}")
                    st.write(f"文件路径：{source.get('source')}")
                    st.caption(source.get("content_preview", ""))
            else:
                st.info("本次回答没有返回知识库来源。")
        else:
            st.error("请求失败")
            st.write(response.status_code)
            st.write(response.text)


st.subheader("已入库文档")

try:
    docs_response = requests.get(
        f"{API_BASE_URL}/documents",
        timeout=30,
    )

    if docs_response.ok:
        docs = docs_response.json()

        if docs:
            st.dataframe(docs, use_container_width=True)
        else:
            st.info("当前还没有入库文档。")
    else:
        st.warning("暂时无法获取已入库文档。")

except requests.exceptions.RequestException:
    st.warning("暂时无法连接后端，请先启动 FastAPI 服务。")