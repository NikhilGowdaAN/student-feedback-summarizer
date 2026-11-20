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
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from app.core import (
    LoRALinear,
    inject_lora,
    get_trainable_parameters,
    train_lora,
    save_lora_weights,
    load_lora_weights,
    generate_summary,
    milestone_2,
)


@pytest.fixture
def mini_gpt2_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=len(tok),
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    return model, tok


def test_milestone_2_adapter_injection_and_freezing(mini_gpt2_and_tokenizer):
    model, _ = mini_gpt2_and_tokenizer
    adapters = inject_lora(model, r=4, lora_alpha=8, target_modules=["c_attn"])

    assert len(adapters) > 0, "Adapters should be injected into target layers"

    # Verify that base parameters are frozen and only LoRA parameters require grad
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA param {name} should be trainable"
        else:
            assert param.requires_grad is False, f"Base model param {name} should be frozen"

    stats = get_trainable_parameters(model)
    assert stats["trainable_params"] > 0
    assert stats["trainable_percent"] < 5.0, "Trainable params should be a small fraction"


def test_milestone_2_zero_init_equivalence(mini_gpt2_and_tokenizer):
    model, tok = mini_gpt2_and_tokenizer
    model.eval()

    dummy_input = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        baseline_logits = model(dummy_input).logits.clone()

    # Inject LoRA
    inject_lora(model, r=4, lora_alpha=8, target_modules=["c_attn"])
    model.eval()

    with torch.no_grad():
        lora_init_logits = model(dummy_input).logits

    # Because lora_B is initialized to 0, initial delta is 0
    diff = torch.max(torch.abs(baseline_logits - lora_init_logits)).item()
    assert diff < 1e-5, f"At initialization, LoRA model should match base model exactly: diff={diff}"


def test_milestone_2_merge_and_unmerge(mini_gpt2_and_tokenizer):
    model, _ = mini_gpt2_and_tokenizer
    adapters = inject_lora(model, r=4, lora_alpha=8, target_modules=["c_attn"])
    model.eval()

    # Set non-zero weights in lora_B to test active delta
    for a in adapters.values():
        nn.init.normal_(a.lora_B.weight, std=0.02)

    dummy_input = torch.tensor([[5, 15, 25, 35]], dtype=torch.long)
    with torch.no_grad():
        unmerged_logits = model(dummy_input).logits.clone()

        for a in adapters.values():
            a.merge()
        merged_logits = model(dummy_input).logits.clone()
        merge_diff = torch.max(torch.abs(unmerged_logits - merged_logits)).item()
        assert merge_diff < 1e-5, f"Merged output should match unmerged output: diff={merge_diff}"

        for a in adapters.values():
            a.unmerge()
        unmerged_restored_logits = model(dummy_input).logits.clone()
        unmerge_diff = torch.max(torch.abs(unmerged_logits - unmerged_restored_logits)).item()
        assert unmerge_diff < 1e-5, f"Unmerge should restore original state: diff={unmerge_diff}"


def test_milestone_2_full_pipeline_and_training(mini_gpt2_and_tokenizer, tmp_path, monkeypatch):
    model, tok = mini_gpt2_and_tokenizer
    monkeypatch.chdir(tmp_path)

    res = milestone_2(
        model=model,
        tokenizer=tok,
        r=4,
        lora_alpha=8,
        epochs=1,
        max_steps=2,
        save_adapter=True,
    )

    assert "model" in res
    assert "tokenizer" in res
    assert "training_result" in res
    assert res["training_result"]["total_steps"] == 2
    assert len(res["training_result"]["loss_history"]) == 2

    # Adapter saving & loading verification
    adapter_file = Path(res["adapter_path"])
    assert adapter_file.exists()
    assert adapter_file.stat().st_size > 0

    # Reload weights
    load_lora_weights(res["model"], adapter_file)

    # Test summary generation
    summary = generate_summary(
        model=res["model"],
        tokenizer=res["tokenizer"],
        feedback="The homework assignments were great and very clear.",
        max_new_tokens=15,
        temperature=0.0,
    )
    assert isinstance(summary, str)
    assert len(summary) > 0


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

