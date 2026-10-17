from app import main
from fastapi.testclient import TestClient
from privrag import generate, rag
from privrag.generate import NOT_FOUND


def test_answer_cites_sources(retriever, monkeypatch):
    monkeypatch.setitem(generate.BACKENDS, "fake", lambda q, chunks: "Take reasonable steps [1].")
    out = rag.answer("protect personal information from misuse", retriever, backend="fake")
    assert out["sources"][0]["doc_id"] == "app-11" and not out["citation_inferred"]


def test_low_score_is_not_found(retriever):
    out = rag.answer("capital city of Canada", retriever, backend="extractive")
    assert out == {"answer": NOT_FOUND, "sources": [], "citation_inferred": False}


def test_model_refusal_is_not_found(retriever, monkeypatch):
    monkeypatch.setitem(generate.BACKENDS, "fake", lambda q, chunks: NOT_FOUND)
    assert rag.answer("protect personal information from misuse", retriever, "fake")["answer"] == NOT_FOUND


def test_extractive_backend(retriever):
    out = rag.answer("enter personal information into generative AI tools", retriever, backend="extractive")
    assert "[1]" in out["answer"] and out["sources"][0]["doc_id"] == "ai-commercial-products"


def test_api(retriever, monkeypatch):
    monkeypatch.setattr(main, "retriever", lambda: retriever)
    monkeypatch.setattr(main.config, "LLM_BACKEND", "extractive")
    client = TestClient(main.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "protect personal information from misuse"})
    assert r.status_code == 200 and r.json()["sources"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert "<form" in client.get("/").text
