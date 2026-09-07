# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained RULER helpers for benchmark_ruler.py.

RULER (https://github.com/NVIDIA/RULER) is a synthetic long-context suite —
retrieval, variable tracking, word extraction, QA — generated per target context
length. Loaded from the preprocessed ``simonjegou/ruler`` parquets, whose configs
are the lengths (4096 / 8192 / 16384) and whose ``test`` split holds all 13
tasks. Each row: context, question, answer_prefix (primes the answer, appended
after the prompt), answer (gold items), task, max_new_tokens.

Metrics from NVIDIA/RULER eval/synthetic/constants.py: ``string_match_all``
(recall over gold items) for retrieval / tracking / extraction,
``string_match_part`` (any gold item) for QA. Reported in [0, 1].
"""

from typing import Any

# Available context-length configs in simonjegou/ruler. Order ascending so a
# sweep does the cheap (short-context) configs first.
RULER_LENGTHS: tuple[str, ...] = ("4096", "8192", "16384")

# The 13 RULER tasks (NVIDIA/RULER synthetic suite). Used for the --tasks filter
# and help text; the loader derives the actual set from the data, so this stays
# informational and need not be exhaustive against future dataset revisions.
RULER_TASKS: tuple[str, ...] = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
    "qa_1", "qa_2",
)

_RULER_REPO = "simonjegou/ruler"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_ruler(
    length: str,
    n_data: int | None = None,
    tasks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load one RULER context-length config as normalized sample dicts.

    ``length`` is a config name (e.g. "4096"). ``tasks`` keeps only the listed
    task names (default: all). ``n_data`` caps the number of samples kept *per
    task* (default: all 500), matching how RULER reports per-task accuracy.
    Returns a flat list ordered by (task, original index)."""
    from datasets import load_dataset

    if length not in RULER_LENGTHS:
        raise ValueError(
            f"Unknown RULER length {length!r}. Available: {list(RULER_LENGTHS)}."
        )
    keep = set(tasks) if tasks else None

    samples = load_dataset(_RULER_REPO, length, split="test")

    per_task_count: dict[str, int] = {}
    dataset: list[dict[str, Any]] = []
    for row in samples:
        task = row["task"]
        if keep is not None and task not in keep:
            continue
        if n_data is not None and per_task_count.get(task, 0) >= n_data:
            continue
        per_task_count[task] = per_task_count.get(task, 0) + 1
        dataset.append(
            {
                "context": row["context"],
                "question": row["question"],
                "answer_prefix": row["answer_prefix"],
                "answers": list(row["answer"]),
                "task": task,
                "max_new_tokens": int(row["max_new_tokens"]),
            }
        )

    print(
        f"\n{_RULER_REPO}:{length} loaded, #data: {len(dataset)} "
        f"across {len(per_task_count)} task(s)"
    )
    return dataset


def group_by_task(dataset: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket samples by task name, preserving load order within each bucket."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in dataset:
        grouped.setdefault(sample["task"], []).append(sample)
    return grouped


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _is_harmony_model(model_name: str) -> bool:
    """gpt-oss uses the OpenAI harmony format, whose chat template cannot be made
    to suppress the analysis (chain-of-thought) channel via apply_chat_template
    kwargs (enable_thinking / reasoning_effort have no effect on the generation
    prompt), and whose ``<|start|>assistant`` tail breaks when an answer_prefix is
    appended raw. We detect it and route to the hand-built harmony template
    instead (see build_prompt)."""
    name = model_name.lower()
    return "gpt-oss" in name or "gpt_oss" in name


def build_prompt(tokenizer: Any, model_name: str, sample: dict[str, Any]) -> str:
    """Build the full RULER prompt string for one sample.

    The kvpress/RULER reference protocol: chat-template the user content
    (context + question) with a generation prompt, then append ``answer_prefix``
    so generation continues from it. ``enable_thinking=False`` matches the
    reference (a no-op for the instruct checkpoints used here).

    Two families take the hand-built scbench_local template instead: harmony
    (gpt-oss), whose apply_chat_template cannot suppress the reasoning channel
    and mis-handles the appended prefix, and any model with no chat template."""
    content = sample["context"] + sample["question"]
    answer_prefix = sample["answer_prefix"]

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and not _is_harmony_model(model_name):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Some templates reject unknown kwargs; retry without enable_thinking.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        # Harmony (gpt-oss) or no chat template: reuse the model-family
        # prefix/postfix markers from scbench_local (task "qa" -> generic
        # instruction; gpt-oss postfix opens the final channel).
        from scbench_local import template

        prefix, postfix = template(model_name, "qa")
        prompt = prefix + content + postfix

    return prompt + answer_prefix


# ---------------------------------------------------------------------------
# Metrics (NVIDIA/RULER eval/synthetic/constants.py)
# ---------------------------------------------------------------------------

def _string_match_all(prediction: str, answers: list[str]) -> float:
    """Recall: fraction of gold items appearing (case-insensitive) in pred."""
    if not answers:
        return 0.0
    pred = prediction.lower()
    hits = sum(1.0 for ans in answers if ans.lower() in pred)
    return hits / len(answers)


def _string_match_part(prediction: str, answers: list[str]) -> float:
    """Any: 1.0 if at least one gold item appears in pred, else 0.0."""
    pred = prediction.lower()
    return 1.0 if any(ans.lower() in pred for ans in answers) else 0.0


def metric_name(task: str) -> str:
    """RULER metric used for a task: QA tasks use part-match, the rest recall."""
    return "string_match_part" if task.startswith("qa") else "string_match_all"


def score_answer(task: str, prediction: str, answers: list[str]) -> float:
    """Score one prediction against its gold items with the RULER task metric.
    Returns a value in [0, 1]."""
    if metric_name(task) == "string_match_part":
        return _string_match_part(prediction, answers)
    return _string_match_all(prediction, answers)
