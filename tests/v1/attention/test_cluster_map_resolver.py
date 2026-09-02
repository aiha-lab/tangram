# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the head-group cluster-map resolver, and for it being audible.

Head-group clustering is one of the three techniques this fork exists for.
Without a map the engine still runs, on identity (adjacent-head) grouping -- so
every path that loses the map loses a technique silently unless it says so.
These pin each of those paths: that it falls back, that it says why at ERROR,
and that the reason is left where a later reader can attribute a measurement to
it.

CPU only, no model. Run without the root conftest, which imports a package this
fork does not install:

    python -m pytest --noconftest -q tests/v1/attention/test_cluster_map_resolver.py
"""
import json

import numpy as np
import pytest

from vllm.v1.attention.backends import cluster_map_resolver as resolver

MODEL = "Qwen/Qwen3-4B"
SLUG = "qwen3-4b"
SCORER = "snapkv"
PAGE_GROUP_SIZE = 4
NUM_KV_HEADS = 8
NUM_LAYERS = 2


@pytest.fixture(autouse=True)
def clean_resolver_state(monkeypatch):
    """Each test starts with no recorded fallback and no bundled tree, so a
    real cluster_maps/ directory in the checkout cannot make a test pass."""
    monkeypatch.setattr(resolver, "identity_fallback_reason", None)
    monkeypatch.delenv(resolver._CLUSTER_MAPS_DIR_ENV, raising=False)


@pytest.fixture
def errors(monkeypatch) -> list[str]:
    """The resolver's ERROR lines, rendered. vLLM's loggers do not propagate to
    the root logger caplog installs on, so the call is intercepted instead."""
    recorded: list[str] = []
    monkeypatch.setattr(
        resolver.logger, "error",
        lambda message, *args: recorded.append(message % args))
    return recorded


def write_map(
    tmp_path,
    *,
    scope: str = "per_layer",
    slug: str = SLUG,
    scorer_dir: str = SCORER,
    page_group_size: int = PAGE_GROUP_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    meta_scope: str | None = None,
    meta_slug: str | None = None,
) -> str:
    """Write a bundled map where the resolver looks for one, and point the
    resolver's tree override at it. Returns the map's path."""
    suffix = resolver._SCOPE_FILENAME_SUFFIX[scope]
    map_dir = tmp_path / scorer_dir / slug
    map_dir.mkdir(parents=True, exist_ok=True)
    path = map_dir / f"pg{page_group_size}_r0.3{suffix}.npz"

    members = np.arange(NUM_LAYERS * num_kv_heads, dtype=np.int64)
    # ``meta`` is a JSON blob, the shape read_cluster_map_meta expects.
    meta = json.dumps({
        "cluster_scope": meta_scope if meta_scope is not None else scope,
        "source_model": meta_slug if meta_slug is not None else slug,
        "page_group_size": page_group_size,
        "num_kv_heads": num_kv_heads,
    })
    np.savez(
        path,
        cluster_of=(members // page_group_size).reshape(
            NUM_LAYERS, num_kv_heads),
        column_of=(members % page_group_size).reshape(
            NUM_LAYERS, num_kv_heads),
        page_group_size=np.int64(page_group_size),
        meta=np.frombuffer(meta.encode("utf-8"), dtype=np.uint8),
    )
    return str(path)


def resolve(tmp_path=None, **overrides) -> str | None:
    if tmp_path is not None:
        import os
        os.environ[resolver._CLUSTER_MAPS_DIR_ENV] = str(tmp_path)
    kwargs = dict(
        scorer=SCORER,
        model_name=MODEL,
        page_group_size=PAGE_GROUP_SIZE,
        cluster_map_scope="per_layer",
        num_kv_heads=NUM_KV_HEADS,
        tp_world_size=1,
    )
    kwargs.update(overrides)
    return resolver.resolve_bundled_cluster_map(**kwargs)


# --- The path that finds a map ---------------------------------------------


def test_a_matching_map_resolves_and_records_no_fallback(tmp_path, errors):
    path = write_map(tmp_path)

    assert resolve(tmp_path) == path
    assert resolver.identity_fallback_reason is None
    assert errors == []


def test_a_second_resolution_clears_the_previous_reason(tmp_path, errors):
    """The reason describes the current run, so a resolution that succeeds must
    not leave a stale one behind for a reader to misattribute."""
    resolve(tmp_path, tp_world_size=2)
    assert resolver.identity_fallback_reason is not None

    write_map(tmp_path)
    assert resolve(tmp_path) is not None
    assert resolver.identity_fallback_reason is None


# --- The five paths that lose the technique --------------------------------


def test_tensor_parallelism_falls_back_audibly(tmp_path, errors):
    write_map(tmp_path)

    assert resolve(tmp_path, tp_world_size=2) is None
    assert "TP=1" in resolver.identity_fallback_reason
    assert len(errors) == 1
    assert "NOT active" in errors[0]


def test_unknown_scope_falls_back_audibly(tmp_path, errors):
    assert resolve(tmp_path, cluster_map_scope="nonesuch") is None
    assert "nonesuch" in resolver.identity_fallback_reason
    assert len(errors) == 1


def test_a_scorer_without_a_map_falls_back_audibly(tmp_path, errors):
    """StreamingLLM scores by recency and TOVA is head-uniform, so neither has
    a retention profile to cluster by -- expected, but still a lost technique."""
    assert resolve(tmp_path, scorer="streamingllm") is None
    assert "streamingllm" in resolver.identity_fallback_reason
    assert len(errors) == 1


def test_a_missing_bundled_tree_falls_back_audibly(tmp_path, errors):
    """A wheel install carries no tools/ directory."""
    import os
    os.environ[resolver._CLUSTER_MAPS_DIR_ENV] = str(tmp_path / "absent")

    assert resolve() is None
    assert "no bundled cluster map tree" in resolver.identity_fallback_reason
    assert len(errors) == 1


def test_no_filename_match_falls_back_audibly(tmp_path, errors):
    """The tree exists but holds nothing for this model."""
    write_map(tmp_path, slug="some-other-model")

    assert resolve(tmp_path) is None
    assert "no bundled map" in resolver.identity_fallback_reason
    assert len(errors) == 1


def test_a_metadata_mismatch_falls_back_audibly(tmp_path, errors):
    """The filename matched but the file describes a different run, which is
    the one path where a wrong map would otherwise be loaded."""
    write_map(tmp_path, meta_slug="a-different-model")

    assert resolve(tmp_path) is None
    assert "does not describe this run" in resolver.identity_fallback_reason
    assert len(errors) == 1


# --- The path that is not a fallback --------------------------------------


def test_the_uniform_scope_is_silent(tmp_path, errors):
    """``uniform`` gives every entry the same count, so there is no cross-entry
    comparison for a map to inform. Not a loss, and not reported as one."""
    assert resolve(tmp_path, cluster_map_scope=None) is None
    assert resolver.identity_fallback_reason is None
    assert errors == []


# --- Ambiguity is a warning, not a fallback -------------------------------


def test_multiple_matches_pick_one_without_falling_back(tmp_path, errors):
    """base_ratio is ignored at runtime, so any matching ratio is valid."""
    first = write_map(tmp_path)
    second = tmp_path / SCORER / SLUG / "pg4_r0.5_perlayer.npz"
    import shutil
    shutil.copy(first, second)

    assert resolve(tmp_path) in (first, str(second))
    assert resolver.identity_fallback_reason is None
    assert errors == []


def test_a_map_without_metadata_falls_back_audibly(tmp_path, errors):
    """Older maps carry no provenance, so nothing can confirm the filename
    match was the right map -- treated as a mismatch."""
    path = write_map(tmp_path)
    data = dict(np.load(path, allow_pickle=False))
    del data["meta"]
    np.savez(path, **data)

    assert resolve(tmp_path) is None
    assert "does not describe this run" in resolver.identity_fallback_reason
    assert len(errors) == 1
