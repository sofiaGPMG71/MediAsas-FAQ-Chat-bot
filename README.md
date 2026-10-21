# MediAsas FAQ chatbot

A Python / Streamlit application for asking evidence-grounded questions about one or more uploaded PDFs. Its calm, teal healthcare-inspired interface guides people through adding documents, building an index, and checking page-linked evidence. It uses LangChain document splitting, embedding integrations, LangChain InMemoryVectorStore vector search, a LangChain retriever, and an LLM question-answering chain. Choose **Local Ollama**, **OpenAI API**, or a key-free **Offline retrieval demo** in the sidebar. It answers from provided coverage documents and does not provide medical advice.

## Architecture

PDF bytes → page extraction → LangChain recursive splitter (1,000 characters, 180 overlap) → embeddings → session-local LangChain InMemoryVectorStore → top-4 retriever → evidence-labelled prompt → LLM → answer + source/page/chunk evidence.

For conversational follow-ups, the LLM first rewrites the question into a standalone retrieval query using the last six messages. History is bounded to 6,000 characters. Only retrieved evidence supports factual answers. The offline demo uses lexical hashing vectors and returns an exact excerpt; it is **not a GenAI substitute** and does not resolve conversational references.

## Setup (Python 3.11 recommended)

Extract this project, open a terminal inside its folder, and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run app.py
```

On macOS/Linux, use `source .venv/bin/activate` and `cp .env.example .env`. If PowerShell activation is restricted, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py` directly. Open the URL printed by Streamlit, usually http://localhost:8501. Run from the project directory so Streamlit loads the included theme and upload limit.

### Local model (no API key)

Install [Ollama](https://ollama.com/download), then:

```text
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve
```

If the Ollama desktop application already serves port 11434, skip `ollama serve`. Choose **Local Ollama**. The default URL is `http://localhost:11434`. You can type the name of another installed chat model, for example `qwen3:4b`; the embedding model must support embeddings. Chat and embedding models are separate. Model speed and memory needs depend on your hardware. Local mode sends text to the configured Ollama server; keep its URL local for on-device processing. Cloud-tagged Ollama models may use a remote service.

### API model

Edit `.env` and set `OPENAI_API_KEY` to your own key. Choose **OpenAI API**. Never commit `.env`. The sample file contains no secrets. An OpenAI account with API billing/access to the chosen models is required. Document chunks are sent to OpenAI for embeddings and retrieved context is sent for answer generation. No keys are entered into chat or source code.

### Environment variables

- `OPENAI_API_KEY`: required only for OpenAI mode.
- `OPENAI_CHAT_MODEL`: default `gpt-4o-mini`.
- `OPENAI_EMBEDDING_MODEL`: default `text-embedding-3-small`.
- `OLLAMA_BASE_URL`: default `http://localhost:11434`.
- `OLLAMA_CHAT_MODEL`: default `llama3.2:3b`.
- `OLLAMA_EMBEDDING_MODEL`: default `nomic-embed-text`.

Model names and URLs can also be edited in the sidebar. Restart the app after changing environment variables. Rebuild the knowledge base whenever documents, provider, or model settings change.

## Use and example inputs

1. Choose a provider; for a no-setup walkthrough use Offline retrieval demo.
2. Upload one or more PDFs, or check **Use included MediAsas customer FAQ**.
3. Click **Build knowledge base**. Review document, page, and chunk counts.
4. Ask a question and expand **View retrieved evidence** to inspect the exact text, filename, physical PDF page, and chunk number.
5. Use **Clear conversation** to start a new chat; **Reset documents and index** clears the active collection from session memory. Uploaded file widgets can be cleared using their remove controls.

Example questions for the supplied PDF:

- What is MediAsas? (page 1)
- Can a customer hold more than one MediAsas plan? (page 4)
- Is there a lifetime limit? (page 8)
- What waiting periods apply? (page 16)
- Can I claim treatment received outside Malaysia? (page 16)
- Are premiums guaranteed? (page 19)
- Follow-up in LLM modes: Does that change as I get older?
- Unsupported: What is the CEO's favourite pizza topping? (should decline)
- Unsupported precise detail: What exact premium will I pay next year? (should explain that the document does not supply an individual quote)

## Sample data and submission contents

`sample_data/MediAsas_Pilot_FAQs_Customers.pdf` is the original user-supplied 22-page **Version 1: Pilot Programme, 29 July 2026** FAQ, copied unchanged. Its terms describe that pilot version and may change; the assistant does not verify later policies. Ownership remains with the document owner. Included for evaluation; confirm redistribution rights before publishing it in a public repository. No Part B agent workflow is implemented: this submission follows the PDF RAG option (Part A).

- `app.py`: upload controls, provider selection, session state, chat and evidence UI.
- `rag.py`: validation, extraction, chunking, provider adapters, vectors and QA chain.
- `requirements.txt`: tested direct dependencies; `requirements-lock.txt`: complete tested environment.
- `tests/`: ingestion, retrieval, refusal, provider and UI checks.
- `RAG_EXPLANATION.md`: 300–500 word flow explanation.
- `screenshots/`: actual working application captures and verification notes.
- `.env.example`, `.gitignore`, `.streamlit/config.toml`: configuration.

## Grounding, privacy and limitations

The prompt instructs the model to use evidence only, treat document instructions as untrusted, refuse missing answers, and attach source labels. Generated answers without valid retrieved source identifiers are replaced by an explicit fallback. Citations validate identifiers, **not semantic truth**: an LLM can still misread a chunk or cite an unsupported claim. Review original pages, especially tables with multiple columns. The top-4 retriever can miss relevant facts; try a more specific question. No OCR, image/table understanding, agent tools, persistent database, authentication, or production access controls are included. Files/index/history live in the individual Streamlit session; they are not saved to disk by this app. They disappear on session loss/server restart. Do not expose an unauthenticated deployment containing sensitive documents.

Limits: 10 documents, 50 MB combined, 300 pages per document, 2,000 characters per question. Empty, encrypted and invalid PDFs are rejected; image-only pages generate OCR warnings. These are practical demo limits, not comprehensive protection against malicious PDF files. Offline vectors use lexical overlap and may retrieve irrelevant passages; generated modes use learned semantic embeddings.

## Tests and troubleshooting

```text
python -m pytest -q
```

For an Ollama connection error, check `ollama list`, confirm both model names and run `ollama serve` if needed. If using Docker, localhost refers to the container; configure an accessible host URL. For API errors, check `.env`, quota, model access and network connectivity. For slow answers, use a smaller local model or a faster machine. For no text, OCR the PDF outside this app and re-upload. Changing providers intentionally disables chat until the collection is rebuilt, preventing mismatched embedding dimensions or stale evidence. Errors shown in the UI omit provider response bodies to avoid exposing credentials.

Framework references: [Streamlit chat API](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input), [LangChain embeddings](https://docs.langchain.com/oss/python/integrations/embeddings), [LangChain Ollama integration](https://docs.langchain.com/oss/python/integrations/providers/ollama/).
