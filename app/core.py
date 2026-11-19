"""Your code. One stub per milestone; build them in order and keep each one small."""

"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
from pathlib import Path
from typing import Any, Optional

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
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


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
