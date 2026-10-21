from pathlib import Path
import sys
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import KnowledgeBase, extract, models, UNKNOWN

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def files():
    return [("faq.pdf", (ROOT / "sample_data/MediAsas_Pilot_FAQs_Customers.pdf").read_bytes())]


@pytest.fixture(scope="module")
def kb(files):
    return KnowledgeBase.build(files, "Offline retrieval demo", {})


def test_metadata_and_overlap(files):
    pages, chunks, warnings = extract(files)
    assert len(pages) == 22 and len(chunks) > len(pages)
    assert all(1 <= d.metadata["page"] <= 22 for d in chunks)
    assert all(d.metadata["source"] == "faq.pdf" for d in chunks)
    assert len({d.metadata["chunk"] for d in chunks}) == len(chunks)
    assert "MediAsas" in pages[0].page_content


def test_multiple_documents(files):
    pages, _, _ = extract(files + [("second.pdf", files[0][1])])
    assert len(pages) == 44
    assert {p.metadata["source"] for p in pages} == {"faq.pdf", "second.pdf"}


def test_invalid_pdf():
    with pytest.raises(ValueError, match="Cannot read"):
        extract([("bad.pdf", b"not a pdf")])


def test_empty_collection():
    with pytest.raises(ValueError, match="No extractable"):
        extract([])


def test_count_limit(files):
    with pytest.raises(ValueError, match="at most 10"):
        extract(files * 11)


def test_retrieval_and_citation(kb):
    answer, sources = kb.ask("Is there a lifetime limit?", [])
    assert "lifetime" in answer.lower()
    assert "[S1]" in answer
    assert any(s["page"] == 8 for s in sources)


def test_offline_refusal(kb):
    answer, sources = kb.ask("quasar spaceship pizza", [])
    assert answer == UNKNOWN and sources == []


@pytest.mark.parametrize("output", ["It costs a million.", "It costs a million [S99]."])
def test_generated_invalid_citations_rejected(kb, output):
    original = kb.llm
    try:
        kb.llm = FakeListChatModel(responses=[output])
        assert kb.ask("What is MediAsas?", [])[0] == UNKNOWN
    finally:
        kb.llm = original


def test_generated_chain(kb):
    original = kb.llm
    try:
        kb.llm = FakeListChatModel(responses=["MediAsas is a standalone medical plan [S1]."])
        assert "[S1]" in kb.ask("What is MediAsas?", [])[0]
    finally:
        kb.llm = original


def test_missing_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        models("OpenAI API", {"api_key": ""})


def test_question_limit(kb):
    with pytest.raises(ValueError):
        kb.ask("x" * 2001, [])
