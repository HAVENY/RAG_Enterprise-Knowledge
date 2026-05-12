import streamlit as st
import requests


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="Enterprise Knowledge Base", layout="wide")
st.title("Enterprise Knowledge Base")
st.caption("上传企业文档，并通过 RAG 问答快速检索内部知识。")


with st.sidebar:
    st.header("文档上传")
    uploaded_file = st.file_uploader("请选择要上传的文档",  type=["txt", "pdf", "docx", "md"],)
    if uploaded_file and st.button("上传并入库"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "text/plain")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        if response.ok:
            st.success(response.json()["message"])
        else:
            st.error(response.text)


st.subheader("知识问答")
question = st.text_area("输入你的问题")

if st.button("开始提问"):
    if not question.strip():
        st.warning("请先输入问题。")
    else:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=120,
        )

        if response.ok:
            result = response.json()

            st.subheader("回答")
            st.write(result.get("answer", ""))

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
            st.error("请求失败")
            st.write(response.status_code)
            st.write(response.text)


st.subheader("已入库文档")
docs_response = requests.get(f"{API_BASE_URL}/documents", timeout=30)
if docs_response.ok:
    docs = docs_response.json()
    if docs:
        st.dataframe(docs, use_container_width=True)
    else:
        st.info("当前还没有入库文档。")
else:
    st.warning("暂时无法连接后端，请先启动 FastAPI 服务。")
