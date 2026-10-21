from pathlib import Path
import hashlib
import os
import streamlit as st
from dotenv import load_dotenv
from rag import KnowledgeBase

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
st.set_page_config(
    page_title="MediAsas | Coverage FAQ",
    page_icon=":material/health_and_safety:",
    layout="wide",
)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("kb", None)
st.session_state.setdefault("signature", None)

with st.sidebar:
    st.markdown("## :material/health_and_safety: MediAsas")
    st.caption("A clearer guide to your coverage documents")
    provider = st.selectbox(
        "Answer style",
        ["Offline retrieval demo", "Local Ollama", "OpenAI API"],
        help="Choose how answers are prepared. You can change this any time.",
    )
    config = {}
    if provider == "OpenAI API":
        st.caption("Reads OPENAI_API_KEY from .env. PDF text is sent to OpenAI for embeddings and answers.")
        config = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": st.text_input("Chat model", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
            "embedding": st.text_input(
                "Embedding model",
                os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            ),
        }
    elif provider == "Local Ollama":
        st.caption("Runs on your Ollama server. No API key is needed; install both models first.")
        config = {
            "url": st.text_input(
                "Ollama URL",
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            "model": st.text_input(
                "Chat model",
                os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
            ),
            "embedding": st.text_input(
                "Embedding model",
                os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            ),
        }
    else:
        st.caption("No API key or model download. Finds relevant passages without generating answers.")

    st.subheader("Your documents", icon=":material/folder_open:")
    st.caption("Add coverage PDFs to make their information searchable.")
    with st.container(border=True):
        uploads = st.file_uploader(
            "Add PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Text-based PDFs work best. You can add more than one document.",
        )
        sample = st.checkbox(
            "Include the MediAsas sample FAQ",
            value=False,
            help="Use the included customer FAQ as a ready-to-try example.",
        )
    files = [(f.name, f.getvalue()) for f in uploads or []]
    if sample:
        files.append(("MediAsas_Pilot_FAQs_Customers.pdf", (ROOT / "sample_data/MediAsas_Pilot_FAQs_Customers.pdf").read_bytes()))
    digest = hashlib.sha256(repr((provider, config)).encode())
    for name, data in files:
        digest.update(name.encode()); digest.update(data)
    signature = digest.hexdigest()
    ready = st.session_state.kb is not None and st.session_state.signature == signature
    if st.button("Build knowledge base", type="primary", width="stretch", disabled=not files):
        st.session_state.kb = None
        st.session_state.messages = []
        try:
            with st.spinner("Extracting pages, splitting text and building vectors…"):
                st.session_state.kb = KnowledgeBase.build(files, provider, config)
            st.session_state.signature = signature
            ready = True
        except Exception as exc:
            st.error(f"Indexing failed ({type(exc).__name__}). Check PDF validity, model names, API key or Ollama connection. See README troubleshooting.")
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
    if st.button("Reset documents and index", width="stretch"):
        st.session_state.kb = None; st.session_state.messages = []; ready = False
    st.caption("Up to 10 PDFs · 50 MB total · 300 pages per PDF.")

st.badge(
    "Coverage document assistant",
    icon=":material/health_and_safety:",
    color="green",
)
st.title("Health cover, made easier to understand")
st.write(
    "Ask a question about your coverage documents and see the source passages "
    "behind each answer."
)
st.caption(
    "For document information only — this assistant does not provide medical advice "
    "or replace guidance from your insurer or care team."
)

if provider == "Offline retrieval demo":
    st.info(
        "Offline demo: shows relevant passages from your PDFs rather than generated "
        "answers. Choose Local Ollama or OpenAI API for conversational answers.",
        icon=":material/info:",
    )

if ready:
    kb = st.session_state.kb
    st.badge("Ready to chat", icon=":material/check_circle:", color="green")
    with st.container(border=True):
        cols = st.columns(3)
        cols[0].metric("Documents", len(files))
        cols[1].metric("Pages indexed", kb.pages)
        cols[2].metric("Searchable passages", kb.chunks)
    for warning in kb.warnings:
        st.warning(warning)
    st.subheader("Ask about your documents", icon=":material/chat:")
    st.caption("Try one question at a time. You can ask a follow-up in AI answer modes.")
elif st.session_state.kb is not None:
    st.warning(
        "Your documents or answer settings changed. Rebuild the knowledge base "
        "before asking another question.",
        icon=":material/refresh:",
    )
else:
    st.subheader("Get started in three steps", icon=":material/rocket_launch:")
    steps = st.columns(3)
    with steps[0]:
        with st.container(border=True):
            st.markdown("**1 · Choose an answer style**")
            st.caption("Use the offline demo or connect an AI provider.")
    with steps[1]:
        with st.container(border=True):
            st.markdown("**2 · Add your PDFs**")
            st.caption("Upload a coverage document or include the sample FAQ.")
    with steps[2]:
        with st.container(border=True):
            st.markdown("**3 · Build and ask**")
            st.caption("Build your index, then ask a question in the chat.")
    st.caption(
        "Example questions: What is covered? · Are there waiting periods? · "
        "What should I confirm with my insurer?"
    )

def show_sources(sources):
    if sources:
        with st.expander(
            f"View source passages · {len(sources)}",
            icon=":material/menu_book:",
        ):
            for s in sources:
                st.markdown(f"**[{s['label']}] {s['source']} · Page {s['page']} · Chunk {s['chunk']}**")
                st.text(s["text"])

for message in st.session_state.messages:
    avatar = (
        ":material/person:"
        if message["role"] == "user"
        else ":material/health_and_safety:"
    )
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

if question := st.chat_input(
    "Ask about your coverage documents…",
    disabled=not ready,
    max_chars=2000,
):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    try:
        with st.chat_message(
            "assistant",
            avatar=":material/health_and_safety:",
        ):
            with st.spinner("Finding evidence…"):
                answer, sources = st.session_state.kb.ask(question, st.session_state.messages)
            st.markdown(answer)
            show_sources(sources)
        st.session_state.messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer, "sources": sources}])
    except Exception as exc:
        st.error(f"Answer generation failed ({type(exc).__name__}). Check your provider connection, quota and model availability, then retry.")

st.caption(
    "Answers are based on retrieved passages, which may be incomplete or misread. "
    "Check the cited pages and confirm important coverage details with your insurer."
)
