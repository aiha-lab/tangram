# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained SCBench helpers for benchmark_scbench.py.

Replaces the former runtime dependency on FastKVZip's ``prefill`` package so
the benchmark runs without cloning that repo. The dataset loading, prompt
templates, generation-length presets, and answer metrics here are adapted
from FastKVZip (https://github.com/Janghyun1230/FastKVzip), which in turn
adapts the SCBench metrics from Microsoft MInference
(https://github.com/microsoft/MInference/tree/main/scbench).

Provides:
    load_dataset_all, get_data_list, template, set_gen_length,
    evaluate_answer, f1_score
"""

import re
import string
from collections import Counter

# ---------------------------------------------------------------------------
# Dataset name groups (FastKVZip/prefill/eval.py: get_data_list)
# ---------------------------------------------------------------------------

_SHORT = ["squad", "gsm"]
_MID = [
    "scbench_many_shot",
    "scbench_mf",
    "scbench_choice_eng",
    "scbench_qa_eng",
    "scbench_repoqa",
]
_LONG = [
    "scbench_kv",
    "scbench_prefix_suffix",
    "scbench_summary",
    "scbench_vt",
]
_MULTI = [
    "scbench_summary_with_needles",
    "scbench_repoqa_and_kv",
]

# Approximate per-prompt token cost. get_data_list orders an expanded group by
# this ascending so cheap datasets finish first (trends appear early) and the
# heavy long-context tasks (summary/vt/full kv) run last. Order-only; values are
# rough measured prompt lengths.
_DATASET_COST = {
    "gsm": 2_000,
    "squad": 3_000,
    "scbench_kv_short": 18_000,
    "scbench_prefix_suffix_short": 19_000,
    "scbench_choice_eng": 35_000,
    "scbench_mf_mid": 45_000,
    "scbench_qa_eng": 50_000,
    "scbench_repoqa": 65_000,
    "scbench_many_shot": 80_000,
    "scbench_mf": 90_000,
    "scbench_prefix_suffix": 112_000,
    "scbench_summary": 113_000,
    "scbench_vt": 124_000,
    "scbench_kv": 169_000,
}

# Datasets at/above this per-prompt cost form the "slow" phase; below = "fast".
_PHASE_THRESHOLD = 100_000


def get_data_list(
    dataname: str, modelname: str = "", max_model_len: int | None = None
) -> list[str]:
    """Expand a group name (short/mid/long/multi/all) into dataset names.

    A non-group name is returned as a single-element list.

    Variant selection is **capacity-based**, not model-family based: the long
    string-retrieval tasks ship full variants whose prompts reach ~120-170k
    tokens (tokenizer-dependent), so when the model's context window
    (``max_model_len``) cannot hold them we substitute the bundled ``_short``
    variants. Any small-context model (llama, gpt-oss, gemma, ...) thus gets the
    right variant automatically — no per-family special-casing."""
    # "fast"/"slow" are cost-phase slices of the full all-group (split below by
    # _PHASE_THRESHOLD) so a sweep can process every model's cheap datasets first
    # and the heavy long-context ones in a later pass.
    phase = None
    if dataname in ("fast", "slow"):
        phase, dataname = dataname, "all"

    if dataname == "short":
        data_list = list(_SHORT)
    elif dataname == "mid":
        data_list = list(_MID)
    elif dataname == "long":
        data_list = list(_LONG)
    elif dataname == "multi":
        data_list = list(_MULTI)
    elif dataname == "all":
        data_list = _LONG + _SHORT + _MID
    else:
        data_list = [dataname]

    # Full scbench_kv / scbench_prefix_suffix prompts exceed ~120k tokens; fall
    # back to the _short variants when the context window is below ~150k. Models
    # with a larger window (e.g. qwen3 at 262144, qwen2.5-1m) keep the full task.
    if max_model_len is not None and max_model_len < 150_000:
        data_list = [
            f"{x}_short" if x in ("scbench_kv", "scbench_prefix_suffix") else x
            for x in data_list
        ]

    # gemma-3 historically uses the mid-difficulty Math.Find variant; preserve.
    if any(k in modelname.lower() for k in ("gemma3", "gemma-3")):
        data_list = [f"{x}_mid" if x == "scbench_mf" else x for x in data_list]

    # Cheap-first ordering so trends appear early; heavy long-context tasks last.
    data_list = sorted(data_list, key=lambda d: _DATASET_COST.get(d, 70_000))

    # Phase slice: fast = cheap datasets, slow = heavy long-context ones.
    if phase == "fast":
        data_list = [d for d in data_list if _DATASET_COST.get(d, 70_000) < _PHASE_THRESHOLD]
    elif phase == "slow":
        data_list = [d for d in data_list if _DATASET_COST.get(d, 70_000) >= _PHASE_THRESHOLD]

    print(data_list)
    return data_list


# ---------------------------------------------------------------------------
# Prompt templates (FastKVZip/prefill/model/template.py: template)
# ---------------------------------------------------------------------------

def template(model_name: str, task: str) -> tuple[str, str]:
    """Return (prefix, postfix) wrapping a context+question prompt for the
    given model family and task."""
    model_name = model_name.lower()

    if "llama" in model_name or model_name == "duo":
        prefix = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        postfix = (
            "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_name.startswith("qwen"):
        prefix = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        postfix = "<|im_end|>\n<|im_start|>assistant\n"
        if "qwen3-" in model_name and "instruct" not in model_name:
            postfix += "<think>\n\n</think>\n\n"
    elif model_name.startswith("gemma3") or model_name.startswith("gemma-3"):
        prefix = "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n"
        postfix = "<end_of_turn>\n<start_of_turn>model\n"
    elif model_name.startswith("gpt-oss") or model_name.startswith("gpt_oss"):
        # OpenAI harmony format. The postfix opens the assistant turn directly in
        # the *final* channel, which suppresses the analysis (chain-of-thought)
        # channel so the generation is the answer itself — matching the baseline
        # FastKVzip-gpt-oss prompt (prefill/model/template.py) and keeping answer
        # extraction clean. Without this, gpt-oss falls to the generic fallback
        # below and rambles as a base completion, collapsing reasoning/QA scores.
        prefix = "<|start|>system<|message|>You are a helpful assistant.<|end|>"
        prefix += "<|start|>user<|message|>"
        postfix = "<|end|><|start|>assistant<|channel|>final<|message|>"
    else:
        print(
            "**Warning** No prompt template for this model; using a generic "
            "fallback (see scbench_local.template)."
        )
        prefix = "<|begin_of_text|>"
        postfix = "\n\nAnswer: "

    if task.startswith("gsm"):
        prefix += (
            "Given the context, answer to the following reasoning "
            "question.\n\n"
        )
    else:
        prefix += (
            "Given the context, answer to the following question or request "
            "without explanation.\n\n"
        )

    return prefix, postfix


# ---------------------------------------------------------------------------
# Generation length presets (FastKVZip/prefill/utils/func.py: set_gen_length)
# ---------------------------------------------------------------------------

def set_gen_length(dataname: str) -> int:
    """Default max output tokens per dataset."""
    if any(k in dataname for k in ("needle", "_mf")):
        max_len = 48
    elif "prefix_suffix" in dataname:
        max_len = 128
    elif any(k in dataname for k in ("squad", "summary")):
        max_len = 256
    elif any(k in dataname for k in ("gsm", "repoqa")):
        max_len = 512
    else:
        max_len = 96
    print(f"set generation length: {max_len} (see scbench_local.set_gen_length)")
    return max_len


# ---------------------------------------------------------------------------
# Dataset loading (FastKVZip/prefill/data/load.py: load_dataset_all)
# ---------------------------------------------------------------------------

def _check_scbench_name(name: str) -> None:
    tag = name.split("scbench_")[1]
    possible_tags = [
        "many_shot", "mf", "repoqa", "choice_eng", "prefix_suffix",
        "summary", "qa_eng", "vt", "kv", "summary_with_needles",
        "repoqa_and_kv",
    ]
    for suffix in ("_tiny", "_short", "_mid"):
        if suffix.strip("_") in tag:
            tag = tag.split(suffix)[0]
            break
    assert tag in possible_tags, f"SCBench data name does not exist: {name!r}"


def _load_scbench(name: str) -> list[dict]:
    from datasets import load_dataset

    _check_scbench_name(name)
    samples = load_dataset(
        "Jang-Hyun/SCBench-preprocessed",
        data_files=f"{name}.parquet",
        split="train",
    )

    dataset = []
    for data in samples:
        d = {"context": data["prompts"][0], "question": data["prompts"][1:]}
        answers = []
        for gt in data["ground_truth"]:
            answers.append(", ".join(gt) if isinstance(gt, list) else str(gt))
        d["answers"] = answers
        dataset.append(d)
    return dataset


def _load_squad(n_data: int) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset("rajpurkar/squad", split="train")
    pool: dict[str, int] = {}
    contexts: list[str] = []
    questions: list[list[str]] = []
    answers: list[list[str]] = []
    for d in data:
        ctx = d["context"]
        if ctx not in pool:
            pool[ctx] = len(contexts)
            contexts.append(ctx)
            questions.append([d["question"]])
            answers.append(list(d["answers"]["text"]))
        else:
            idx = pool[ctx]
            questions[idx].append(d["question"])
            answers[idx].append(d["answers"]["text"][0])
        if len(pool) > n_data:
            break
    return [
        {"context": c, "question": q, "answers": a}
        for c, q, a in zip(contexts, questions, answers)
    ]


def _load_gsm(tokenizer, n_data: int) -> list[dict]:
    from datasets import load_dataset

    dataset_full = load_dataset("openai/gsm8k", "main", split="test")
    dataset = []
    for data in dataset_full:
        st = data["question"].split(". ")
        context = ". ".join(st[:-1]).strip() + "."
        if len(tokenizer.encode(context, add_special_tokens=False)) < 72:
            continue
        dataset.append(
            {
                "context": context,
                "question": [st[-1].strip()],
                "answers": [data["answer"]],
            }
        )
        if len(dataset) == n_data:
            break
    return dataset


def load_dataset_all(name: str, tokenizer, n_data: int = 100) -> list[dict]:
    """Load a dataset as a list of {context, question[list], answers[list]}."""
    if name == "squad":
        dataset = _load_squad(n_data)
    elif name == "gsm":
        dataset = _load_gsm(tokenizer, n_data)
    elif "scbench" in name:
        dataset = _load_scbench(name)
    else:
        raise ValueError(
            f"Unsupported dataset {name!r}. Supported: scbench_*, squad, gsm."
        )
    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset


# ---------------------------------------------------------------------------
# Answer metrics (FastKVZip/prefill/results/metric.py, from MInference SCBench)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def replace_num(text):
        word_to_number = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        }
        pattern = re.compile(
            r"\b(" + "|".join(word_to_number.keys()) + r")\b"
        )
        return pattern.sub(lambda x: word_to_number[x.group()], text)

    return replace_num(
        white_space_fix(remove_articles(remove_punc(s.lower())))
    )


def f1_score(pred: str, ref: str, normalize: bool = True) -> float:
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    prediction_tokens = pred.split()
    ground_truth_tokens = ref.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


_ROUGE_SCORER = None


def _rouge_score(prediction: str, ground_truth: str) -> float:
    # Lazy: only summary-family tasks need ROUGE. Uses google-research's
    # ``rouge-score`` (the package vLLM already pins in requirements/test.txt),
    # not the similarly named ``rouge`` package.
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    try:
        # RougeScorer.score(target, prediction) — target is the reference.
        return _ROUGE_SCORER.score(ground_truth, prediction)["rougeL"].fmeasure
    except Exception:
        return 0.0


def _include_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def _include_score_multi(pred, ref, normalize=True):
    refs = ref.split(", ")
    if normalize:
        pred = normalize_answer(pred)
        refs = [normalize_answer(r) for r in refs]
    scores = [r in pred for r in refs]
    return sum(scores) / len(scores)


def _include_score_gsm(pred, ref, normalize=True):
    ref = ref.strip().split("#### ")[-1]
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def _option_letter(text: str) -> str | None:
    """Multiple-choice option selected by a many_shot answer.

    SCBench many_shot questions list options ``(A) ... (B) ...`` and the answer
    is one option. Returns the first ``(X)`` with a single A-Z letter (the
    format the in-context examples and answers use), or — when the string is a
    bare single letter (the reference is stored as e.g. ``D``) — that letter.
    Upper-cased; ``None`` when no option letter is present.
    """
    match = re.search(r"\(([A-Za-z])\)", text)
    if match is not None:
        return match.group(1).upper()
    stripped = text.strip()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()
    return None


def _include_score_manyshot(pred, ref, normalize=True):
    """Score a many_shot multiple-choice answer by its selected option letter.

    The reference is an option letter (``D`` or ``(D) Named Entities``); a
    correct prediction selects the same option, emitted as ``(X)``. We compare
    the extracted option letters exactly. The previous bare-substring test
    (``ref in pred``) was wrong: a single-letter reference like ``E`` occurs in
    almost any English text, so it scored essentially every answer correct.
    Falls back to normalized inclusion only when the reference has no option
    letter (not expected for many_shot).
    """
    ref_letter = _option_letter(ref)
    if ref_letter is not None:
        pred_letter = _option_letter(pred)
        return float(pred_letter is not None and pred_letter == ref_letter)
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return float(ref in pred)


def _exact_match_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return pred == ref


def evaluate_answer(
    preds, refs, dataname, fmt, similarity=False, subtask=None
) -> list[float]:
    """Score predictions against references with the SCBench per-task metric.

    The RepoQA structured-similarity path (repo_qa_utils) is intentionally not
    bundled; callers should pass ``similarity=True`` for repoqa datasets (as
    benchmark_scbench.py does), which routes to token-level F1."""
    if "repoqa" in dataname and not similarity:
        raise NotImplementedError(
            "RepoQA structured scoring is not bundled in scbench_local; pass "
            "similarity=True to score repoqa via F1."
        )

    score: list[float] = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if pred.endswith("</s>"):
            pred = pred[:-4]
        if len(pred.strip()) == 0:
            score.append(0.0)
            continue

        name = subtask[i] if subtask is not None else dataname

        if similarity:
            score.append(f1_score(pred, ref))
        elif fmt != "qa":
            score.append(_rouge_score(pred, ref))
        elif "_vt" in name:
            score.append(_include_score_multi(pred, ref, normalize=False))
        elif "_mf" in name:
            score.append(_exact_match_score(pred, ref, normalize=False))
        elif "_many_shot" in name:
            score.append(_include_score_manyshot(pred, ref))
        elif "summary" in name:
            score.append(_rouge_score(pred, ref))
        elif "qa_eng" in name:
            score.append(max(f1_score(pred, ref), _include_score(pred, ref)))
        elif "choice_eng" in name:
            score.append(_include_score(pred.split("\n")[0], ref))
        elif "gsm" in name:
            pred = pred.strip().lower().split("the answer is ")[-1]
            score.append(_include_score_gsm(pred, ref, normalize=False))
        else:
            score.append(_include_score(pred, ref))
    return [float(s) for s in score]
