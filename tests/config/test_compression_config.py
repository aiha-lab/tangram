# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm/config/compression.py -- the ragged-paging and KV-compression
startup validation.

These run on CPU with no model or CUDA. Run them without the root conftest,
which imports a package this fork does not install:

    python -m pytest --noconftest -q tests/config/test_compression_config.py
"""
import os

import pytest

from vllm.config.cache import CacheConfig

BACKEND_ENV = "VLLM_ATTENTION_BACKEND"


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch):
    """Each test decides what the backend environment variable holds, and no
    test leaks its choice into the next one."""
    monkeypatch.delenv(BACKEND_ENV, raising=False)


# --- Constructing a config must not reach outside itself -------------------


def test_cache_config_leaves_the_backend_env_alone():
    """A dataclass constructor must not write process-global state.

    CacheConfig is constructed more than once per process (standalone, from
    EngineArgs, in tests) and at a point where nothing has decided which
    backend the run uses. Choosing one here reaches every later reader of the
    variable, including code that has nothing to do with this config object.
    """
    CacheConfig(page_group_size=4)

    assert BACKEND_ENV not in os.environ


def test_vllm_config_defaults_the_backend_for_ragged_paging():
    """Ragged paging is implemented only in FlashAttention, so assembling a
    config with it enabled selects that backend -- at assembly, where the other
    cross-config decisions about ragged paging are already made."""
    from vllm.config.vllm import VllmConfig

    VllmConfig(cache_config=CacheConfig(page_group_size=4))

    assert os.environ[BACKEND_ENV] == "FLASH_ATTN"


def test_vllm_config_keeps_an_explicit_backend():
    """An explicit choice wins: the default exists because the user made none."""
    from vllm.config.vllm import VllmConfig

    os.environ[BACKEND_ENV] = "TRITON_ATTN"
    VllmConfig(cache_config=CacheConfig(page_group_size=4))

    assert os.environ[BACKEND_ENV] == "TRITON_ATTN"


def test_vllm_config_leaves_the_backend_alone_without_ragged_paging():
    """No ragged paging, no reason to constrain the backend."""
    from vllm.config.vllm import VllmConfig

    VllmConfig(cache_config=CacheConfig(page_group_size=None))

    assert BACKEND_ENV not in os.environ


# --- The rules the validators enforce --------------------------------------


def test_ratio_out_of_range_is_rejected():
    """The ratio is the evicted fraction, so 1.0 would evict everything; it is
    also the on/off switch, so an out-of-range value must not read as "off"."""
    with pytest.raises(ValueError, match="0 <= r < 1"):
        CacheConfig(compression_ratio=1.0)


def test_the_two_retention_targets_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CacheConfig(compression_ratio=0.5, compression_budget_tokens=4096)


def test_budget_below_the_unconditionally_kept_positions_is_rejected():
    """Sink plus protected tail are kept whatever the scores say, so a budget
    at or below their sum can never be reached."""
    with pytest.raises(ValueError, match="kept unconditionally"):
        CacheConfig(compression_budget_tokens=64, compression_chunk_size=2048)


def test_forced_score_source_without_a_budget_is_rejected():
    """Forcing a source is an ablation instrument; under the ratio regime no
    cached position is ever rescored, so it would silently do nothing."""
    with pytest.raises(ValueError, match="requires"):
        CacheConfig(compression_slot_score_source="persist")


def test_recompute_needs_a_scorer_that_rescores_the_cache():
    with pytest.raises(ValueError, match="scores cached positions"):
        CacheConfig(
            compression_budget_tokens=8192,
            compression_slot_score_source="recompute",
            compression_scorer="snapkv",
        )


def test_unknown_scorer_is_rejected():
    with pytest.raises(ValueError, match="compression_scorer must be one of"):
        CacheConfig(compression_ratio=0.3, compression_scorer="nonesuch")


def test_unknown_budget_scope_is_rejected():
    with pytest.raises(ValueError, match="compression_budget_scope must be one of"):
        CacheConfig(compression_ratio=0.3, compression_budget_scope="nonesuch")


def test_page_group_size_must_divide_the_kv_heads():
    """A head group is a page, so a group that straddles the KV heads has no
    layout. The counts arrive from the model, not the constructor, so the check
    runs again once they are populated."""
    from vllm.config.compression import validate_extended_fields

    config = CacheConfig(page_group_size=3)
    config.num_kv_heads = 8
    config.num_hidden_layers = 2

    with pytest.raises(ValueError, match="must be divisible"):
        validate_extended_fields(config)


def test_compression_requires_ragged_paging():
    with pytest.raises(ValueError, match="requires page_group_size"):
        CacheConfig(compression_ratio=0.3, page_group_size=None)


# --- Prefix caching -------------------------------------------------------


def test_prefix_caching_is_disabled_and_the_warning_says_how_to_keep_it(monkeypatch):
    """Prefix caching cannot represent ragged paging's per-(layer, group) block
    layout, so enabling either turns it off.

    It stays a warning rather than an error: page_group_size defaults to 4, so
    raising would fail startup for everyone who passes --enable-prefix-caching
    without also turning ragged paging off. The warning therefore has to name
    the setting that would keep prefix caching, or the reader is told what was
    taken away and not how to get it back.

    The warning is read off the logger call rather than through caplog: vLLM's
    loggers do not propagate to the root logger caplog installs on.
    """
    from vllm.config import compression

    warnings: list[str] = []
    monkeypatch.setattr(
        compression.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    config = CacheConfig(page_group_size=4, enable_prefix_caching=True)

    assert config.enable_prefix_caching is False
    assert any("--page-group-size=None" in text for text in warnings), warnings


def test_prefix_caching_survives_without_ragged_paging():
    config = CacheConfig(page_group_size=None, enable_prefix_caching=True)

    assert config.enable_prefix_caching is True


# --- The model allowlist ---------------------------------------------------


def test_unvalidated_architecture_is_rejected():
    """An unsupported model does not fail loudly on the ragged path -- it falls
    back to dense attention and returns wrong output -- so it is rejected at
    startup."""
    with pytest.raises(ValueError, match="does not support Tangram"):
        CacheConfig(page_group_size=4).verify_model_support("MambaForCausalLM")


def test_every_advertised_architecture_passes():
    """The README's "Supported Models" table is the promise; the allowlist is
    what enforces it, so a model advertised there and missing here would fail
    at startup. Checked together to keep the two from drifting."""
    for architecture in (
        "LlamaForCausalLM",  # meta-llama/Llama-3.1-8B-Instruct
        "Qwen3ForCausalLM",  # Qwen/Qwen3-4B-Instruct-2507
        "Qwen3MoeForCausalLM",  # Qwen/Qwen3-30B-A3B-Instruct-2507
        "Gemma3ForConditionalGeneration",  # google/gemma-3-12b-it
        "GptOssForCausalLM",  # openai/gpt-oss-20b
    ):
        CacheConfig(page_group_size=4).verify_model_support(architecture)


def test_allowlist_is_not_consulted_without_the_feature():
    """Plain vLLM runs any model this fork has not validated."""
    CacheConfig(page_group_size=None).verify_model_support("MambaForCausalLM")
