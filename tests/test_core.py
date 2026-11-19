"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs.
from pathlib import Path
from app.core import (
    milestone_1,
    format_prompt,
    get_dataset,
    save_dataset,
    SYNTHETIC_FEEDBACK_DATA,
)
from app.storage import Store


def test_milestone_1_structure_and_types():
    data = milestone_1(save_to_disk=False)
    assert isinstance(data, list)
    assert len(data) >= 20

    for item in data:
        assert isinstance(item, dict)
        assert "id" in item and isinstance(item["id"], int)
        assert "feedback" in item and isinstance(item["feedback"], str)
        assert "summary" in item and isinstance(item["summary"], str)
        assert "category" in item and isinstance(item["category"], str)
        assert "sentiment" in item and isinstance(item["sentiment"], str)
        assert "actionable_takeaway" in item and isinstance(item["actionable_takeaway"], str)

        # Content checks
        assert len(item["feedback"].strip()) >= 30, "Feedback should be substantive"
        assert len(item["summary"].strip()) >= 10, "Summary should be informative"
        # Concise check: summary should be shorter than feedback
        assert len(item["summary"]) < len(item["feedback"]), "Summary must be concise relative to feedback"


def test_milestone_1_diversity():
    data = milestone_1(save_to_disk=False)
    categories = {item["category"] for item in data}
    sentiments = {item["sentiment"] for item in data}

    # Verify comprehensive category coverage across course dimensions
    assert len(categories) >= 8, f"Expected broad topical diversity, found {len(categories)}"
    expected_categories = {"lecture_clarity", "pacing", "assignments", "exams", "grading", "teaching_assistants"}
    assert expected_categories.issubset(categories)

    # Verify sentiment diversity (positive, negative, constructive)
    assert {"positive", "negative", "constructive"}.issubset(sentiments)


def test_milestone_1_filtering_and_formatting():
    # Category filtering
    lectures = get_dataset(category="lecture_clarity")
    assert len(lectures) >= 2
    assert all(item["category"] == "lecture_clarity" for item in lectures)

    # Sentiment filtering
    positive = get_dataset(sentiment="positive")
    assert len(positive) >= 5
    assert all(item["sentiment"] == "positive" for item in positive)

    # Prompt formatting
    sample = SYNTHETIC_FEEDBACK_DATA[0]
    prompt_full = format_prompt(sample, include_summary=True)
    assert sample["feedback"] in prompt_full
    assert sample["summary"] in prompt_full
    assert "### Student Feedback:" in prompt_full
    assert "### Summary:" in prompt_full

    prompt_infer = format_prompt(sample, include_summary=False)
    assert sample["feedback"] in prompt_infer
    assert sample["summary"] not in prompt_infer


def test_milestone_1_persistence(tmp_path, monkeypatch):
    test_json = tmp_path / "custom_data.json"
    saved_path = save_dataset(test_json)
    assert saved_path.exists()
    assert len(saved_path.read_text(encoding="utf-8")) > 500

    # Test full milestone_1 save to disk and store
    monkeypatch.chdir(tmp_path)
    res = milestone_1(save_to_disk=True)
    assert (tmp_path / "data" / "synthetic_feedback.json").exists()

    store = Store()
    stored_data = store.get("synthetic_feedback")
    assert stored_data is not None
    assert len(stored_data) == len(res)


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate the quality of generated summaries against original feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

