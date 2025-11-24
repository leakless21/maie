import pytest

from src.processors.llm.processor import LLMProcessor


VIETNAMESE_PHRASE = (
    "Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ những video hấp dẫn"
)


def test_loads_hallu_phrase():
    p = LLMProcessor(model_path="dummy")
    norm = p._normalize_text(VIETNAMESE_PHRASE)
    assert norm in p._llm_hallu_phrases


def test_remove_exact_hallu_removes_exact_match():
    p = LLMProcessor(model_path="dummy")
    # Exact phrase becomes empty after filtering
    out = p._remove_exact_hallu(VIETNAMESE_PHRASE)
    assert out == ""

    # Partial phrase is unchanged
    out2 = p._remove_exact_hallu("Hãy subscribe cho kênh khác")
    assert out2 == "Hãy subscribe cho kênh khác"


def test_postprocess_summary_strips_hallu_from_data():
    p = LLMProcessor(model_path="dummy")
    # title exactly equal to hallu phrase should be removed
    data = {"title": VIETNAMESE_PHRASE, "participants": ["Alice", VIETNAMESE_PHRASE]}
    transcript = "Alice talked about many things"

    result = p._postprocess_summary("meeting_notes_v1", transcript, data)

    assert result["title"] is None
    assert result["participants"] == ["Alice"]
