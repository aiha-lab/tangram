# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gpt-oss on SCBench with reasoning allowed.

gpt-oss is a reasoning model, evaluated by OpenAI with its analysis channel
open. With ``reasoning_effort`` set, the harness must build exactly the prompt
the model's own chat template builds (``Reasoning: <effort>`` in the system
message, the assistant turn left open so the model picks the channel), and must
score only the final channel. Without it, every model keeps today's prompt.
"""
import os
import re
import sys

import pytest

TANGRAM_BENCH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "tangram")
sys.path.insert(0, TANGRAM_BENCH)

import benchmark_scbench as bs  # noqa: E402
import scbench_local as sl  # noqa: E402

GPT_OSS = "/raid/LLM/gpt-oss-20b"
DATASET = "scbench_repoqa"
SAMPLE = [{"context": "CTX", "question": ["Q1", "Q2"], "answers": ["a1", "a2"]}]


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(GPT_OSS):
        pytest.skip(f"{GPT_OSS} not present")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GPT_OSS)


def _turns(tokenizer, model_name, reasoning_effort):
    out = bs.build_multi_turn_from_dataset(
        tokenizer=tokenizer, model_name=model_name, dataset=SAMPLE,
        dataset_name=DATASET, start_idx=0, end_idx=1, max_questions=None,
        max_tokens=64, reasoning_effort=reasoning_effort)
    return [tokenizer.decode(t) for t in out[1][0]]


def test_reasoning_prompt_matches_chat_template(tokenizer):
    turns = _turns(tokenizer, "gpt-oss-20b", "low")

    prefix, _ = sl.template("gpt-oss-20b", DATASET)
    instruction = prefix.split("<|start|>user<|message|>")[-1]
    content = instruction + "CTX" + "\n\nQ: Q1"
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, reasoning_effort="low")
    # The template stamps today's date; the harness pins one for reproducibility.
    expected = re.sub(r"Current date: \d{4}-\d{2}-\d{2}",
                      f"Current date: {sl.HARMONY_DATE}", expected)

    assert turns[0] + turns[1] == expected


def test_reasoning_followup_turn_opens_user_message(tokenizer):
    turns = _turns(tokenizer, "gpt-oss-20b", "low")

    assert turns[2] == "<|start|>user<|message|>\n\nQ: Q2<|end|><|start|>assistant"


def test_followup_turn_unchanged_without_reasoning(tokenizer):
    turns = _turns(tokenizer, "gpt-oss-20b", None)

    assert turns[2] == "\n\nQ: Q2<|end|><|start|>assistant<|channel|>final<|message|>"


def test_final_channel_is_scored_alone():
    raw = ("<|channel|>analysis<|message|>887 or 987? 887.<|end|>"
           "<|start|>assistant<|channel|>final<|message|>887<|return|>")
    assert sl.harmony_final(raw) == ("887", True)

    truncated = "<|channel|>analysis<|message|>still thinking"
    assert sl.harmony_final(truncated) == ("", False)


@pytest.mark.parametrize("model_name", ["Qwen3-4B-Instruct-2507", "llama3.1-8b-instruct"])
def test_other_families_ignore_reasoning(model_name):
    assert sl.template(model_name, DATASET, reasoning_effort="low") == \
        sl.template(model_name, DATASET)


def test_gpt_oss_without_reasoning_keeps_final_channel():
    _, postfix = sl.template("gpt-oss-20b", DATASET)
    assert postfix.endswith("<|start|>assistant<|channel|>final<|message|>")
