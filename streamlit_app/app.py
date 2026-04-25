# DEPLOY LÊN STREAMLIT COMMUNITY CLOUD:
# 1. Chạy local: streamlit run streamlit_app/app.py
# 2. Upload tài liệu qua sidebar để index vào db_streamlit/
# 3. git add db_streamlit/ streamlit_app/ && git commit && git push
# 4. Vào share.streamlit.io → New app → chọn repo
#    Main file path: streamlit_app/app.py
# 5. Advanced settings → Secrets: GOOGLE_API_KEY = "your_key_here"
#
# LƯU Ý: db_streamlit/ phải được commit lên GitHub để Streamlit Cloud
# có dữ liệu đã index (filesystem của Streamlit Cloud bị reset khi redeploy).

import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

load_dotenv(Path(__file__).parent.parent / ".env")

DB_PATH = Path(__file__).parent.parent / "db_streamlit"
COLLECTION_NAME = "slide_streamlit"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 10

st.set_page_config(
    page_title="Document Chat",
    page_icon="📚",
    layout="wide",
)


def get_api_key() -> str | None:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    return st.session_state.get("api_key") or None


@st.cache_resource
def get_vectorstore(api_key: str) -> Chroma:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    embedding = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(DB_PATH),
    )


def chunk_file(uploaded_file) -> list[Document]:
    filename = uploaded_file.name
    docs: list[Document] = []

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(Document(
                page_content=f"[{filename} - Trang {page_num}]\n{text}",
                metadata={"filename": filename, "page": page_num, "source": filename},
            ))
    else:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        chunk_size = 2000
        for i, start in enumerate(range(0, len(content), chunk_size), start=1):
            chunk = content[start:start + chunk_size].strip()
            if not chunk:
                continue
            docs.append(Document(
                page_content=f"[{filename} - Phần {i}]\n{chunk}",
                metadata={"filename": filename, "page": i, "source": filename},
            ))

    return docs


def index_documents(files, api_key: str):
    all_docs: list[Document] = []
    for f in files:
        chunks = chunk_file(f)
        all_docs.extend(chunks)

    if not all_docs:
        st.warning("Không tìm thấy nội dung trong các file đã upload.")
        return

    vs = get_vectorstore(api_key)
    batch_size = 5
    total = len(all_docs)
    progress = st.progress(0, text="Đang index tài liệu...")

    for i in range(0, total, batch_size):
        batch = all_docs[i:i + batch_size]
        vs.add_documents(batch)
        progress.progress(
            min((i + batch_size) / total, 1.0),
            text=f"Đang index... {min(i + batch_size, total)}/{total} chunks",
        )

    progress.empty()
    st.success(f"✅ Đã index {total} chunks từ {len(files)} file!")
    st.session_state.pop("indexed_files_cache", None)


def get_indexed_files(api_key: str) -> list[dict]:
    if "indexed_files_cache" in st.session_state:
        return st.session_state["indexed_files_cache"]
    try:
        vs = get_vectorstore(api_key)
        result = vs._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result.get("metadatas", []):
            fname = meta.get("filename", "unknown")
            counts[fname] = counts.get(fname, 0) + 1
        file_list = [{"filename": k, "chunks": v} for k, v in counts.items()]
        st.session_state["indexed_files_cache"] = file_list
        return file_list
    except Exception:
        return []


def delete_document(filename: str, api_key: str):
    vs = get_vectorstore(api_key)
    vs._collection.delete(where={"filename": filename})
    st.session_state.pop("indexed_files_cache", None)


def get_sources_and_context(query: str, api_key: str) -> tuple[str, list[dict]]:
    vs = get_vectorstore(api_key)
    docs = vs.similarity_search(query, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in docs)
    seen: set[tuple] = set()
    sources: list[dict] = []
    for doc in docs:
        fname = doc.metadata.get("filename", "")
        page = doc.metadata.get("page", 0)
        key = (fname, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": fname,
                "page": page,
                "preview": doc.page_content[:200],
            })
    return context, sources


def stream_answer(query: str, context: str, api_key: str):
    client = genai.Client(api_key=api_key)
    prompt = (
        "Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh tài liệu "
        "được cung cấp. Nếu không tìm thấy thông tin liên quan, hãy nói rõ điều đó. "
        "Trả lời bằng tiếng Việt.\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {query}"
    )
    for chunk in client.models.generate_content_stream(model=LLM_MODEL, contents=prompt):
        if chunk.text:
            yield chunk.text


# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Document Understanding")
    st.divider()

    api_key = get_api_key()
    if not api_key:
        st.markdown("**🔑 Google API Key**")
        entered = st.text_input(
            "API Key",
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if entered:
            st.session_state["api_key"] = entered
            api_key = entered
    else:
        st.success("🔑 API Key đã được cấu hình")

    st.divider()
    st.markdown("**📂 Tài liệu**")

    uploaded_files = st.file_uploader(
        "Upload tài liệu",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if api_key:
            if st.button("⚡ Index tài liệu", use_container_width=True, type="primary"):
                with st.spinner("Đang xử lý..."):
                    index_documents(uploaded_files, api_key)
        else:
            st.warning("Vui lòng nhập API Key trước.")

    if api_key:
        st.divider()
        st.markdown("**📋 Đã index**")
        indexed = get_indexed_files(api_key)
        if indexed:
            for item in indexed:
                col1, col2 = st.columns([5, 1])
                col1.caption(f"📄 **{item['filename']}** ({item['chunks']} chunks)")
                if col2.button("🗑️", key=f"del_{item['filename']}", help="Xóa tài liệu"):
                    delete_document(item["filename"], api_key)
                    st.rerun()
        else:
            st.caption("Chưa có tài liệu nào.")

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("💬 Hỏi đáp tài liệu")

if not st.session_state.messages:
    st.info("👋 Hãy upload tài liệu ở sidebar rồi đặt câu hỏi!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Nguồn tham khảo"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src['filename']} — Trang {src['page']}")
                    st.text(src.get("preview", "")[:200])

user_prompt = st.chat_input("Nhập câu hỏi...")

if user_prompt:
    if not api_key:
        st.warning("⚠️ Vui lòng nhập Google API Key ở sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu..."):
            context, sources = get_sources_and_context(user_prompt, api_key)

        response = st.write_stream(stream_answer(user_prompt, context, api_key))

        if sources:
            with st.expander("📄 Nguồn tham khảo"):
                for src in sources:
                    st.caption(f"📄 {src['filename']} — Trang {src['page']}")
                    st.text(src.get("preview", "")[:200])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
