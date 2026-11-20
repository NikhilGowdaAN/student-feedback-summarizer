"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from app.storage import Store


# Synthetic dataset of realistic student feedback and summary pairs
SYNTHETIC_FEEDBACK_DATA: list[dict[str, Any]] = [
    {
        "id": 1,
        "feedback": "Professor Smith broke down complex dynamic programming concepts using intuitive recursion trees and real-world caching examples. The live-coding demonstrations during lecture made edge-case handling much clearer than reading the textbook alone.",
        "summary": "Clear explanations of dynamic programming with effective recursion trees and live coding demonstrations.",
        "category": "lecture_clarity",
        "sentiment": "positive",
        "actionable_takeaway": "Continue incorporating live-coding demos and visual recursion trees in algorithmic lectures."
    },
    {
        "id": 2,
        "feedback": "The lectures were mostly the instructor reading equations directly off the slides without explaining the intuition behind gradient descent proofs. When students asked for clarification, we were just told to refer to chapter 4.",
        "summary": "Lectures lacked conceptual intuition and relied too heavily on reading equation-dense slides.",
        "category": "lecture_clarity",
        "sentiment": "negative",
        "actionable_takeaway": "Focus lecture time on conceptual explanations and address student questions directly instead of deferring to textbook."
    },
    {
        "id": 3,
        "feedback": "The pace of the first three weeks on basic Python was agonizingly slow, but then in week four we suddenly jumped directly into PyTorch convolutional backpropagation with almost no transition or explanation.",
        "summary": "Uneven course pacing with slow introductory weeks followed by an abrupt jump into complex deep learning.",
        "category": "pacing",
        "sentiment": "negative",
        "actionable_takeaway": "Smooth out curriculum pacing by shortening basic syntax review and providing intermediate neural network tutorials."
    },
    {
        "id": 4,
        "feedback": "The progression from single-variable calculus to multivariate optimization was very well-paced. The weekly review checkpoints before starting new topics helped ensure everyone was on the same page.",
        "summary": "Well-paced curriculum progression with helpful weekly review checkpoints.",
        "category": "pacing",
        "sentiment": "positive",
        "actionable_takeaway": "Maintain weekly review checkpoints as an effective rhythm for mathematical topics."
    },
    {
        "id": 5,
        "feedback": "The machine learning homework assignments were intellectually stimulating, but the starter code had several hidden environment dependency bugs that took hours to debug before we could even start the actual algorithm implementation. Please provide a verified Docker or Conda environment.",
        "summary": "Stimulating assignments were hampered by dependency bugs in the starter code.",
        "category": "assignments",
        "sentiment": "constructive",
        "actionable_takeaway": "Provide verified containerized development environments (Docker/Conda) for all programming assignments."
    },
    {
        "id": 6,
        "feedback": "Building our own mini search engine in Assignment 2 was the best project I've done in college. The autograder gave instant feedback on test cases, which made debugging memory leaks straightforward.",
        "summary": "Engaging search engine project with helpful instant autograder feedback.",
        "category": "assignments",
        "sentiment": "positive",
        "actionable_takeaway": "Retain the mini search engine assignment and expand the autograder test case suite."
    },
    {
        "id": 7,
        "feedback": "Homework 4 was assigned on Monday and due on Friday, yet it required implementing a distributed Paxos consensus algorithm from scratch. The workload for this single assignment exceeded 35 hours and conflicted with midterm exams in other core classes.",
        "summary": "Unreasonable 35-hour workload for the Paxos assignment within a tight 4-day window.",
        "category": "assignments",
        "sentiment": "negative",
        "actionable_takeaway": "Extend deadline windows to at least two weeks for complex systems programming assignments."
    },
    {
        "id": 8,
        "feedback": "The midterm exam did not reflect the practice problems or homework assignments at all. While the assignments focused on practical Python implementations and data analysis, the midterm was 80% rote memorization of theoretical theorem proofs.",
        "summary": "Midterm exam misaligned with practical course material, focusing heavily on proof memorization.",
        "category": "exams",
        "sentiment": "negative",
        "actionable_takeaway": "Align exam questions with practical homework assignments rather than pure proof memorization."
    },
    {
        "id": 9,
        "feedback": "The final exam was tough but very fair. The questions tested our conceptual understanding of database normalization and SQL query optimization rather than trick questions or arbitrary syntax.",
        "summary": "Challenging and fair final exam testing conceptual database concepts rather than syntax trivia.",
        "category": "exams",
        "sentiment": "positive",
        "actionable_takeaway": "Keep designing exams focused on conceptual problem solving and schema optimization."
    },
    {
        "id": 10,
        "feedback": "The TAs were absolute lifesavers during office hours. Alex stayed an extra 45 minutes on Thursday night to help students debug memory segmentation faults in their C memory allocator projects.",
        "summary": "Dedicated TAs provided exceptional debugging assistance during office hours.",
        "category": "teaching_assistants",
        "sentiment": "positive",
        "actionable_takeaway": "Recognize TA performance and ensure office hours have sufficient staffing during systems projects."
    },
    {
        "id": 11,
        "feedback": "Office hours queue times regularly exceeded two hours on days leading up to assignment deadlines, and several TAs seemed unfamiliar with the autograder test cases or gave contradictory advice on how to structure the graph algorithms.",
        "summary": "Excessive office hours wait times and contradictory guidance from unprepared TAs.",
        "category": "teaching_assistants",
        "sentiment": "negative",
        "actionable_takeaway": "Conduct pre-assignment briefing sessions for TAs and introduce an organized queue system."
    },
    {
        "id": 12,
        "feedback": "The interactive Jupyter notebooks provided alongside each lecture were fantastic. Being able to modify hyperparameters and visualize decision boundaries in real time cemented the theoretical concepts.",
        "summary": "Interactive Jupyter notebooks effectively reinforced theoretical lecture concepts.",
        "category": "course_materials",
        "sentiment": "positive",
        "actionable_takeaway": "Continue developing interactive notebooks with visual components for each core topic."
    },
    {
        "id": 13,
        "feedback": "The required textbook is expensive and uses outdated Python 2 code snippets in its examples. The instructor's lecture slides are much more current and sufficient; the course syllabus should make the textbook optional.",
        "summary": "Required textbook is outdated and costly; instructor slides are sufficient.",
        "category": "course_materials",
        "sentiment": "constructive",
        "actionable_takeaway": "Switch to open-source modern materials or make the legacy textbook optional."
    },
    {
        "id": 14,
        "feedback": "Grading rubrics were not released prior to project submissions, and grades for Assignment 1 were not returned until week 9 of the semester. Without feedback on earlier assignments, it was impossible to know if we were repeating the same design mistakes in later submissions.",
        "summary": "Delayed grading feedback and absent rubrics prevented students from learning from earlier mistakes.",
        "category": "grading",
        "sentiment": "negative",
        "actionable_takeaway": "Publish clear rubrics upfront and commit to a 2-week turnaround time for grading."
    },
    {
        "id": 15,
        "feedback": "The grading was transparent and prompt. Detailed comments on our pull requests helped us learn professional software engineering and clean code practices.",
        "summary": "Prompt, transparent grading with high-quality pull request feedback.",
        "category": "grading",
        "sentiment": "positive",
        "actionable_takeaway": "Standardize inline PR code reviews across all assignments."
    },
    {
        "id": 16,
        "feedback": "The course description lists introductory linear algebra as recommended rather than mandatory, but eigenvalues and SVD were assumed knowledge from day one. Future students should be strongly warned or provided with a week zero refresher module.",
        "summary": "Understated math prerequisites; recommends a week-zero linear algebra refresher module.",
        "category": "prerequisites",
        "sentiment": "constructive",
        "actionable_takeaway": "Add a prerequisite self-assessment quiz and optional week-zero math bootcamp."
    },
    {
        "id": 17,
        "feedback": "The weekly reading load of 80 pages combined with two coding labs and a weekly quiz made this 3-credit course consume more time than my other three courses combined. It feels like a 6-credit workload.",
        "summary": "Course workload is excessively high and disproportionate for a 3-credit class.",
        "category": "workload",
        "sentiment": "negative",
        "actionable_takeaway": "Audit course workload to trim non-essential reading and consolidate weekly deliverables."
    },
    {
        "id": 18,
        "feedback": "The workload was steady and predictable. Having assignments due every other Sunday gave us adequate time to digest material without experiencing end-of-semester burnout.",
        "summary": "Balanced and predictable biweekly assignment schedule supported healthy study habits.",
        "category": "workload",
        "sentiment": "positive",
        "actionable_takeaway": "Preserve the biweekly deadline cadence in future course iterations."
    },
    {
        "id": 19,
        "feedback": "The open-ended group project was a great opportunity to explore transformer models, but team formation was completely unmoderated. Our group ended up with two members who never contributed, and there was no peer evaluation mechanism to address free-riders.",
        "summary": "Group project lacked team accountability and peer evaluation to prevent free-riders.",
        "category": "group_projects",
        "sentiment": "constructive",
        "actionable_takeaway": "Introduce mandatory mid-project and final peer evaluations with individual grade weighting."
    },
    {
        "id": 20,
        "feedback": "The milestone check-ins for the capstone project kept our group disciplined throughout the term. The industry mentor feedback during week 8 was invaluable for refining our database schema.",
        "summary": "Structured milestones and industry mentor feedback significantly enriched the capstone project.",
        "category": "group_projects",
        "sentiment": "positive",
        "actionable_takeaway": "Continue partnering with industry mentors for intermediate capstone feedback."
    },
    {
        "id": 21,
        "feedback": "Recorded lectures lacked closed captions, and the audio quality was often muffled when the professor walked away from the podium mic. Adding accurate captions and a lapel mic would greatly help non-native English speakers and hard-of-hearing students.",
        "summary": "Muffled lecture audio and missing closed captions hinder accessibility.",
        "category": "accessibility",
        "sentiment": "constructive",
        "actionable_takeaway": "Adopt lapel microphones and enable automated captions on all recorded lecture videos."
    },
    {
        "id": 22,
        "feedback": "The remote GPU cluster crashed repeatedly during the 48 hours before the final project was due. Many students couldn't complete their model training through no fault of their own, but no deadline extension was granted.",
        "summary": "Pre-deadline server crashes disrupted project training without appropriate extensions.",
        "category": "infrastructure",
        "sentiment": "negative",
        "actionable_takeaway": "Establish clear infrastructure failover contingency plans and flexible extension policies."
    },
    {
        "id": 23,
        "feedback": "Using GitHub Classroom and automated CI/CD checks for homework submissions made the workflow seamless and mirrored modern developer environments.",
        "summary": "GitHub Classroom and automated CI/CD checks provided a smooth and professional workflow.",
        "category": "infrastructure",
        "sentiment": "positive",
        "actionable_takeaway": "Maintain GitHub Classroom and automated CI test suites for all programming courses."
    },
    {
        "id": 24,
        "feedback": "The course covers extensive relational database theory like relational algebra and normal forms, but never touched on NoSQL, vector databases, or real-world cloud data warehousing, which are essential for modern software engineering.",
        "summary": "Curriculum is overly theoretical and misses modern databases like NoSQL and vector stores.",
        "category": "curriculum_relevance",
        "sentiment": "constructive",
        "actionable_takeaway": "Modernize the database syllabus to include modern cloud warehouses and vector databases."
    },
    {
        "id": 25,
        "feedback": "Great balance between mathematical foundations and hands-on coding. We implemented logistic regression from scratch in NumPy first, and then learned how to use Scikit-Learn efficiently.",
        "summary": "Strong blend of fundamental mathematics and practical library implementation.",
        "category": "curriculum_relevance",
        "sentiment": "positive",
        "actionable_takeaway": "Maintain the pedagogical sequence of scratch implementation followed by high-level libraries."
    },
    {
        "id": 26,
        "feedback": "Professor Patel's passion for distributed systems is contagious. She brought current research papers and real incidents from major cloud outages into class discussions, making every session feel relevant.",
        "summary": "Enthusiastic instruction connecting real-world cloud outages to distributed systems theory.",
        "category": "instructor_engagement",
        "sentiment": "positive",
        "actionable_takeaway": "Encourage instructors to ground theoretical topics in recent industry case studies."
    },
    {
        "id": 27,
        "feedback": "The instructor seemed disengaged and dismissed questions during lecture by saying 'that's beyond the scope of this course.' Discussions on the online forum often went unanswered for days.",
        "summary": "Instructor seemed disengaged, dismissed student questions, and neglected online forum threads.",
        "category": "instructor_engagement",
        "sentiment": "negative",
        "actionable_takeaway": "Foster a more welcoming classroom environment and designate daily forum monitoring shifts."
    },
    {
        "id": 28,
        "feedback": "Announcements about homework deadline changes and office hours cancellations were scattered across Discord, Canvas announcements, and Piazza. Having a single consolidated source of truth for course updates would avoid a lot of confusion.",
        "summary": "Course communications were fragmented across Discord, Canvas, and Piazza.",
        "category": "communication",
        "sentiment": "constructive",
        "actionable_takeaway": "Designate Canvas or a single communication platform as the sole source of official updates."
    }
]


def format_prompt(example: dict[str, Any], include_summary: bool = True) -> str:
    """Format a student feedback and summary pair into a standard prompt for training or inference."""
    prompt = (
        "Below is student feedback for a course. Generate a concise summary of the key points.\n\n"
        f"### Student Feedback:\n{example['feedback'].strip()}\n\n"
        "### Summary:\n"
    )
    if include_summary and "summary" in example and example["summary"]:
        prompt += f"{example['summary'].strip()}"
    return prompt


def get_dataset(
    category: Optional[str] = None,
    sentiment: Optional[str] = None
) -> list[dict[str, Any]]:
    """Retrieve the synthetic feedback dataset, optionally filtered by category or sentiment."""
    data = SYNTHETIC_FEEDBACK_DATA
    if category is not None:
        data = [item for item in data if item["category"] == category]
    if sentiment is not None:
        data = [item for item in data if item["sentiment"] == sentiment]
    return data


def save_dataset(filepath: Optional[str | Path] = None) -> Path:
    """Save the synthetic dataset as a JSON file to disk."""
    out_path = Path(filepath or "data/synthetic_feedback.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(SYNTHETIC_FEEDBACK_DATA, indent=2), encoding="utf-8")
    return out_path


# --- Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs.
# Hint: Focus on creating diverse, representative examples for your dataset.
def milestone_1(save_to_disk: bool = True) -> list[dict[str, Any]]:
    """Prepares and validates the synthetic student feedback and summary dataset.

    Returns:
        list[dict[str, Any]]: The list of feedback-summary pairs with categories, sentiments,
                             and actionable takeaways.
    """
    dataset = get_dataset()

    # Validate non-empty and minimum diversity requirements
    if not dataset or len(dataset) < 10:
        raise ValueError(f"Dataset too small: {len(dataset)} examples found.")

    categories = {item["category"] for item in dataset}
    sentiments = {item["sentiment"] for item in dataset}
    if len(categories) < 3:
        raise ValueError(f"Insufficient category diversity: only {len(categories)} categories.")
    if len(sentiments) < 2:
        raise ValueError(f"Insufficient sentiment diversity: only {len(sentiments)} sentiments.")

    if save_to_disk:
        # Save to raw JSON file
        save_dataset()
        # Also persist to app store
        store = Store()
        store.put("synthetic_feedback", dataset)

    return dataset


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# Hint: Use the `lfllm-lflora` technique to inject and train the LoRA adapter.

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer following the lfllm-lflora technique.

    Wraps a target linear layer (nn.Linear or transformers.pytorch_utils.Conv1D)
    by adding low-rank trainable decomposition matrices A and B:
        output = original_layer(x) + (x @ A^T @ B^T) * (alpha / r)
    """

    def __init__(
        self,
        original_layer: nn.Module,
        r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0
        self.merged = False

        # Support both standard nn.Linear and HuggingFace Conv1D (used in GPT-2)
        if hasattr(original_layer, "in_features"):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
            self.is_conv1d = False
        elif hasattr(original_layer, "weight"):
            self.in_features = original_layer.weight.shape[0]
            self.out_features = original_layer.weight.shape[1]
            self.is_conv1d = True
        else:
            raise ValueError(f"Unsupported layer type for LoRA injection: {type(original_layer)}")

        # Freeze the original layer weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Initialize LoRA decomposition matrices
        # lora_A: in_features -> r (initialized with Kaiming uniform)
        # lora_B: r -> out_features (initialized to zeros so delta is initially 0)
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original_layer(x)
        if self.merged or self.r == 0:
            return orig_out
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return orig_out + lora_out

    def merge(self) -> None:
        """Merge LoRA weights into original layer weights for zero-overhead inference."""
        if self.merged:
            return
        # delta_w = B @ A has shape (out_features, in_features)
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        if self.is_conv1d:
            # Conv1D weights have shape (in_features, out_features)
            self.original_layer.weight.data += delta_w.T
        else:
            self.original_layer.weight.data += delta_w
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from the original layer."""
        if not self.merged:
            return
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        if self.is_conv1d:
            self.original_layer.weight.data -= delta_w.T
        else:
            self.original_layer.weight.data -= delta_w
        self.merged = False


def inject_lora(
    model: nn.Module,
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> dict[str, LoRALinear]:
    """Injects LoRA adapters into target linear modules of a pre-trained model.

    Freezes all base model parameters and replaces target layers with LoRALinear.
    """
    if target_modules is None:
        target_modules = ["c_attn"]

    # Step 1: Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    adapter_modules: dict[str, LoRALinear] = {}

    # Step 2: Traverse modules and replace targets
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) or target in name.split("."):
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]

                orig_layer = getattr(parent, attr_name)
                if not isinstance(orig_layer, LoRALinear):
                    lora_layer = LoRALinear(
                        orig_layer,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(parent, attr_name, lora_layer)
                    adapter_modules[name] = lora_layer

    return adapter_modules


def get_trainable_parameters(model: nn.Module) -> dict[str, Any]:
    """Calculate trainable vs total parameter statistics."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    percent = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": round(percent, 4),
    }


class FeedbackDataset(Dataset):
    """PyTorch Dataset for causal language modeling on student feedback-summary pairs."""

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.encodings: list[dict[str, torch.Tensor]] = []
        for item in data:
            prompt_text = format_prompt(item, include_summary=True)
            # Append EOS token to signal end of generation
            if tokenizer.eos_token and not prompt_text.endswith(tokenizer.eos_token):
                prompt_text += tokenizer.eos_token

            encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask padding tokens so loss is not computed on them
            labels[attention_mask == 0] = -100

            self.encodings.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encodings[idx]


def train_lora(
    model: nn.Module,
    tokenizer: Any,
    dataset: Optional[list[dict[str, Any]]] = None,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 2,
    max_steps: Optional[int] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Trains the injected LoRA adapter on feedback-summary pairs."""
    if dataset is None:
        dataset = get_dataset()

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.train()

    # Build DataLoader
    train_dataset = FeedbackDataset(dataset, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Only train LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Was LoRA injected?")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    loss_history: list[float] = []
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            labels = batch["labels"].to(target_device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            loss_history.append(step_loss)
            epoch_loss += step_loss
            batch_count += 1
            total_steps += 1

            if max_steps is not None and total_steps >= max_steps:
                break

        if max_steps is not None and total_steps >= max_steps:
            break

    model.eval()
    final_loss = loss_history[-1] if loss_history else 0.0
    return {
        "epochs": epoch + 1,
        "total_steps": total_steps,
        "final_loss": round(final_loss, 4),
        "loss_history": [round(x, 4) for x in loss_history],
    }


def save_lora_weights(model: nn.Module, save_path: str | Path = "models/lora_adapter.pt") -> Path:
    """Extracts and saves only the LoRA adapter parameters."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data.cpu()

    torch.save(lora_state_dict, str(out_path))
    return out_path


def load_lora_weights(model: nn.Module, load_path: str | Path = "models/lora_adapter.pt") -> None:
    """Loads previously saved LoRA adapter parameters into the model."""
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA weights not found at: {path}")

    lora_state_dict = torch.load(str(path), map_location="cpu")
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)


def generate_summary(
    model: nn.Module,
    tokenizer: Any,
    feedback: str,
    max_new_tokens: int = 40,
    min_new_tokens: int = 2,
    temperature: float = 0.7,
    device: Optional[str] = None,
) -> str:
    """Generates a concise summary for a student feedback comment."""
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()

    prompt = format_prompt({"feedback": feedback}, include_summary=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0][input_len:]
    summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not summary:
        summary = tokenizer.decode(generated_tokens).strip()
    return summary


def milestone_2(
    model_name: str = "gpt2",
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    epochs: int = 1,
    max_steps: Optional[int] = 5,
    save_adapter: bool = True,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Milestone 2: Injects and trains a LoRA adapter on a pre-trained LLM using lfllm-lflora.

    Args:
        model_name: Base LLM identifier (defaults to 'gpt2').
        r: LoRA decomposition rank.
        lora_alpha: LoRA scaling hyperparameter.
        lora_dropout: Dropout probability inside LoRA module.
        epochs: Number of training epochs.
        max_steps: Optional maximum training steps (useful for quick fine-tuning).
        save_adapter: Whether to save adapter weights to models/lora_adapter.pt.
        model: Optional pre-loaded model (for testing or custom architectures).
        tokenizer: Optional pre-loaded tokenizer.
        device: Device to train on ('cpu', 'cuda').

    Returns:
        dict[str, Any]: Trained model, tokenizer, parameter stats, and training results.
    """
    # 1. Load tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model
    if model is None:
        model = GPT2LMHeadModel.from_pretrained(model_name)

    # 3. Inject LoRA adapters
    adapter_modules = inject_lora(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_attn"],
    )

    # 4. Measure parameter efficiency
    stats = get_trainable_parameters(model)

    # 5. Train LoRA adapter on synthetic feedback dataset
    train_result = train_lora(
        model=model,
        tokenizer=tokenizer,
        epochs=epochs,
        max_steps=max_steps,
        device=device,
    )

    # 6. Save adapter weights
    adapter_path = None
    if save_adapter:
        adapter_path = str(save_lora_weights(model, "models/lora_adapter.pt"))

    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainable_params": stats["trainable_params"],
        "total_params": stats["total_params"],
        "trainable_percent": stats["trainable_percent"],
        "adapter_modules": adapter_modules,
        "training_result": train_result,
        "adapter_path": adapter_path,
    }


# --- Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
# Hint: Design a simple Streamlit interface for inputting feedback and displaying summaries.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate the quality of generated summaries against original feedback.
# Hint: Consider metrics like ROUGE or manual inspection for evaluating summary quality.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Apply the `lfllm-lflora` merging technique to integrate the adapter for efficient inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
