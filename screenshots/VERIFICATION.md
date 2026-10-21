# Demo and verification

Captured from the running Streamlit application on 21 October 2026 using the supplied 22-page FAQ. The extracted collection contains 69 chunks. Screenshots are real browser captures, not mockups.

1. `01_pdf_uploaded_and_indexed.png`: PDF uploaded through the file uploader, knowledge base built, page and chunk counts shown. Offline demo provider.
2. `02_answer_with_page_evidence.png`: lifetime-limit question, retrieved excerpt and expanded source evidence with page numbers. Offline demo provider.
3. `03_missing_answer.png`: unsupported question explicitly declined. Offline demo provider.
4. `04_ollama_generated_answer.png`: full GenAI RAG answer using local `llama3.2:latest` and `nomic-embed-text` embeddings. Source labels trace to retrieved evidence.
5. `05_ollama_missing_answer.png`: unsupported CEO/pizza question declined by the local RAG path.

Automated validation: **13 tests passed** under Python 3.11.15 and the pinned direct dependencies. Tests cover page metadata, multiple PDFs, malformed/empty documents, document-count limit, retrieval, offline refusal, generated citation identifiers (using a fake model), missing API key, question-length limit, and Streamlit chat/rebuild/clear state. A separate live local integration check verified the lifetime-limit answer against page 8 and confirmed refusal of the unsupported CEO question; see `local_integration_check.txt`.

OpenAI adapter construction and missing-key behavior were checked. A paid OpenAI generation call was not run because no user API key was provided. Local Ollama was tested against already-installed models. These checks do not prove universal factual accuracy; assess new questions against the original PDF.

The PDF is copied byte-for-byte from the supplied file. The RAG explanation is approximately 420 words. No API keys, virtual environment, or generated vector data are included in the submission archive.
