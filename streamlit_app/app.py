from __future__ import annotations

# DEPLOY LÊN STREAMLIT COMMUNITY CLOUD:
# 1. Chạy local: streamlit run streamlit_app/app.py
# 2. Upload tài liệu qua sidebar để index vào db_streamlit/
# 3. git add db_streamlit/ streamlit_app/ && git commit && git push
# 4. Vào share.streamlit.io → New app → chọn repo
#    Main file path: streamlit_app/app.py
# 5. Advanced settings → Secrets: GOOGLE_API_KEY = "your_key_here"

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv(Path(__file__).parent.parent / ".env")

DB_PATH = Path(__file__).parent.parent / "db_streamlit"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
#LLM_MODEL = "gemini-2.5-flash-lite"
OCR_MODEL = "gemini-2.5-flash"
TOP_K = 10

st.set_page_config(page_title="Document Chat", page_icon="📚", layout="wide")

# ── Google OAuth Login Gate ───────────────────────────────────────────────────
if not st.user.is_logged_in:
    st.title("📚 Document Chat")
    st.markdown("Vui lòng đăng nhập để sử dụng ứng dụng.")
    if st.button("🔐 Đăng nhập với Google", type="primary", use_container_width=False):
        st.login("google")
    st.stop()


# ── Lightweight vector store (numpy + json, không cần chromadb) ──────────────

@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class SimpleVectorStore:
    def __init__(self, store_path: Path, api_key: str):
        self.store_path = Path(store_path)
        self.api_key = api_key
        self._docs_file = self.store_path / "documents.json"
        self._vecs_file = self.store_path / "vectors.npy"
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._docs: list[dict] = []
        self._vecs: np.ndarray | None = None
        self._load()

    def _load(self):
        if self._docs_file.exists():
            self._docs = json.loads(self._docs_file.read_text(encoding="utf-8"))
        if self._vecs_file.exists() and self._docs:
            self._vecs = np.load(str(self._vecs_file))

    def _save(self):
        self._docs_file.write_text(
            json.dumps(self._docs, ensure_ascii=False),
            encoding="utf-8",
        )
        if self._vecs is not None and len(self._vecs):
            np.save(str(self._vecs_file), self._vecs)
        elif self._vecs_file.exists():
            self._vecs_file.unlink()

    def _embed(self, texts: list[str]) -> np.ndarray:
        client = genai.Client(api_key=self.api_key)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return np.array([e.values for e in result.embeddings], dtype=np.float32)

    def add_documents(self, docs: list[Document]):
        texts = [d.page_content for d in docs]
        new_vecs = self._embed(texts)
        for doc in docs:
            self._docs.append({"content": doc.page_content, "metadata": doc.metadata})
        self._vecs = (
            new_vecs if self._vecs is None or len(self._vecs) == 0
            else np.vstack([self._vecs, new_vecs])
        )
        self._save()

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        if not self._docs or self._vecs is None or len(self._vecs) == 0:
            return []
        q_vec = self._embed([query])[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        norms = np.linalg.norm(self._vecs, axis=1)
        safe_norms = np.where(norms == 0, 1e-10, norms)
        scores = (self._vecs @ q_vec) / (safe_norms * q_norm)
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            Document(
                page_content=self._docs[i]["content"],
                metadata=self._docs[i]["metadata"],
            )
            for i in top_idx
        ]

    def get_all_metadata(self) -> list[dict]:
        return [d["metadata"] for d in self._docs]

    def delete_by_filename(self, filename: str):
        keep = [
            i for i, d in enumerate(self._docs)
            if d["metadata"].get("filename") != filename
        ]
        self._docs = [self._docs[i] for i in keep]
        self._vecs = self._vecs[keep] if keep and self._vecs is not None else None
        self._save()


# ── Helpers ──────────────────────────────────────────────────────────────────

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
def get_store(api_key: str) -> SimpleVectorStore:
    return SimpleVectorStore(DB_PATH, api_key)


def ocr_page_with_gemini(pdf_bytes: bytes, page_index: int, api_key: str) -> str:
    """Render a PDF page to image and OCR it with Gemini Vision."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=OCR_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
                    types.Part(text=(
                        "Trích xuất toàn bộ văn bản từ hình ảnh trang tài liệu này. "
                        "Chỉ trả về văn bản gốc, không thêm giải thích hay định dạng thêm."
                    )),
                ],
            )
        ],
    )
    return (response.text or "").strip()


def chunk_file(uploaded_file, api_key: str | None = None) -> tuple[list[Document], int]:
    """Returns (docs, ocr_page_count)."""
    filename = uploaded_file.name
    docs: list[Document] = []
    ocr_count = 0
    if filename.lower().endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text and api_key:
                try:
                    text = ocr_page_with_gemini(pdf_bytes, page_num - 1, api_key)
                    if text:
                        ocr_count += 1
                except Exception as e:
                    st.error(f"OCR trang {page_num} thất bại: {type(e).__name__}: {e}")
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
    return docs, ocr_count


def index_documents(files, api_key: str):
    all_docs: list[Document] = []
    total_ocr = 0
    for f in files:
        docs, ocr_count = chunk_file(f, api_key)
        all_docs.extend(docs)
        total_ocr += ocr_count

    if not all_docs:
        st.warning("Không tìm thấy nội dung trong các file đã upload.")
        return

    store = get_store(api_key)
    batch_size = 5
    total = len(all_docs)
    progress = st.progress(0, text="Đang index tài liệu...")

    for i in range(0, total, batch_size):
        batch = all_docs[i:i + batch_size]
        store.add_documents(batch)
        progress.progress(
            min((i + batch_size) / total, 1.0),
            text=f"Đang index... {min(i + batch_size, total)}/{total} chunks",
        )

    progress.empty()
    msg = f"✅ Đã index {total} chunks từ {len(files)} file!"
    if total_ocr:
        msg += f" (OCR {total_ocr} trang ảnh)"
    st.success(msg)
    st.session_state.pop("indexed_files_cache", None)


def get_indexed_files(api_key: str) -> list[dict]:
    if "indexed_files_cache" in st.session_state:
        return st.session_state["indexed_files_cache"]
    try:
        store = get_store(api_key)
        counts: dict[str, int] = {}
        for meta in store.get_all_metadata():
            fname = meta.get("filename", "unknown")
            counts[fname] = counts.get(fname, 0) + 1
        file_list = [{"filename": k, "chunks": v} for k, v in counts.items()]
        st.session_state["indexed_files_cache"] = file_list
        return file_list
    except Exception:
        return []


def delete_document(filename: str, api_key: str):
    get_store(api_key).delete_by_filename(filename)
    st.session_state.pop("indexed_files_cache", None)


def get_sources_and_context(query: str, api_key: str) -> tuple[str, list[dict]]:
    docs = get_store(api_key).similarity_search(query, k=TOP_K)
    context = "\n\n".join(d.page_content for d in docs)
    seen: set[tuple] = set()
    sources: list[dict] = []
    for doc in docs:
        key = (doc.metadata.get("filename", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": doc.metadata.get("filename", ""),
                "page": doc.metadata.get("page", 0),
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


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Document Understanding")
    st.divider()

    user = st.user
    st.markdown(f"👤 **{user.name}**")
    st.caption(user.email)
    if st.button("🚪 Đăng xuất", use_container_width=True):
        st.logout()
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

# ── Main ──────────────────────────────────────────────────────────────────────
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
