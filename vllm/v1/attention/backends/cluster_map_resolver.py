# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-resolution of bundled head-group cluster maps.

Finds the ``.npz`` under
``tools/head_group_clustering/cluster_maps/<scorer-dir>/<model-slug>/`` matching
the running model, scorer, ``page_group_size`` and budget scope, so the default
config needs no explicit ``--head-group-cluster-map``. Convention
``pg<page_group_size>_r<ratio>[_perlayer].npz``, the ratio globbed because the
runtime ignores ``base_ratio``.

Only locates and identity-checks; ``ragged_layout.load_cluster_map`` remains the
authoritative validator of the contents. Runs once at config finalization and is
frozen on, so every consumer reads one path.
"""
from __future__ import annotations

import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

#: Override for the bundled cluster-map tree (relocated / vendored deployments).
_CLUSTER_MAPS_DIR_ENV = "TANGRAM_CLUSTER_MAPS_DIR"

#: Bundled-maps location relative to the repo root, used when the env is unset.
_BUNDLED_MAPS_RELPATH = ("tools", "head_group_clustering", "cluster_maps")

#: Scorers that ship no map (clustering by retention is meaningless): recency
#: (StreamingLLM) and head-uniform (TOVA). They fall back to identity.
_SCORERS_WITHOUT_CLUSTER_MAP = frozenset({"streamingllm", "tova"})

#: Scorer -> directory name where it differs (matches build_all_profiles.sh).
_SCORER_DIR_ALIASES = {"expected_attention": "ea"}

#: Cluster-map scope -> filename suffix (cross-layer maps have none).
_SCOPE_FILENAME_SUFFIX = {"global": "", "per_layer": "_perlayer"}

#: Slug overrides for HF ids whose map directory differs from the natural slug.
#: Empty today. Kept separate from the gate's alias table (different layout).
_MODEL_SLUG_ALIASES: dict[str, str] = {}


#: Set by the last :func:`resolve_bundled_cluster_map` that fell back, to the
#: reason it did. ``None`` after a resolution that found a map. Read it to
#: attribute a result: a log line scrolls away, a published number does not.
identity_fallback_reason: str | None = None


def _fall_back_to_identity(reason: str) -> None:
    """Record and announce that head-group clustering is not active.

    Losing clustering changes which information survives an eviction at the same
    total, so a run that falls back is not a slightly different run -- it is a
    different method. Hence ERROR. The engine continues anyway, because identity
    grouping is a valid layout and refusing to start would break the
    configurations that have no map today (tensor parallelism, and any install
    without the bundled tree).
    """
    global identity_fallback_reason
    identity_fallback_reason = reason
    logger.error(
        "Head-group clustering is NOT active: %s. Falling back to identity "
        "(adjacent-head) grouping. Eviction still works, but which positions "
        "survive differs from a clustered run, so do not compare this run's "
        "accuracy against clustered numbers.", reason)


def _bundled_maps_base_dir() -> Path | None:
    """Root of the bundled cluster-map tree, or ``None`` if absent (e.g. a wheel
    install without ``tools/``), in which case resolution degrades to identity.
    """
    override = os.environ.get(_CLUSTER_MAPS_DIR_ENV)
    if override:
        base = Path(override).expanduser()
        return base if base.is_dir() else None

    import vllm

    # vllm.__file__ is <repo>/vllm/__init__.py; parent.parent is the repo root.
    repo_root = Path(vllm.__file__).resolve().parent.parent
    base = repo_root.joinpath(*_BUNDLED_MAPS_RELPATH)
    return base if base.is_dir() else None


def _model_slug(model_name: str) -> str:
    """Map-directory slug for an HF id: lowercased last path component
    (``Qwen/Qwen3-4B-Instruct-2507`` -> ``qwen3-4b-instruct-2507``)."""
    slug = model_name.rstrip("/").split("/")[-1].lower()
    return _MODEL_SLUG_ALIASES.get(slug, slug)


def _scorer_dir(scorer: str) -> str | None:
    """Cluster-map directory name for a scorer, or ``None`` if it ships none."""
    if scorer in _SCORERS_WITHOUT_CLUSTER_MAP:
        return None
    return _SCORER_DIR_ALIASES.get(scorer, scorer)


def _validate_cluster_map_meta(
    path: str,
    *,
    expected_scope: str,
    model_slug: str,
    page_group_size: int,
    num_kv_heads: int | None,
) -> bool:
    """Cross-check an auto-found map's recorded ``meta`` against the model so a
    slug collision can't silently load a shape-compatible but wrong map. Checks
    scope, source model, ``page_group_size``, and ``num_kv_heads``; any mismatch
    (or missing meta) returns ``False`` to fall back to identity. The array
    contents are still validated authoritatively by ``load_cluster_map``."""
    from vllm.v1.attention.backends.ragged_layout import (
        read_cluster_map_meta,
    )

    meta = read_cluster_map_meta(path)
    if meta is None:
        logger.warning(
            "Bundled cluster map %s has no metadata to verify against the "
            "model; falling back to identity grouping.", path)
        return False

    map_scope = meta.get("cluster_scope")
    if map_scope != expected_scope:
        logger.warning(
            "Bundled cluster map %s has cluster_scope=%r but the budget "
            "scope needs %r; falling back to identity grouping.",
            path, map_scope, expected_scope)
        return False

    map_model = meta.get("source_model")
    if map_model is None or _model_slug(map_model) != model_slug:
        logger.warning(
            "Bundled cluster map %s was built for model %r but is being "
            "resolved for slug %r; falling back to identity grouping.",
            path, map_model, model_slug)
        return False

    map_pg = meta.get("page_group_size")
    if map_pg is not None and int(map_pg) != page_group_size:
        logger.warning(
            "Bundled cluster map %s has page_group_size=%d but the runtime "
            "uses %d; falling back to identity grouping.",
            path, int(map_pg), page_group_size)
        return False

    map_heads = meta.get("num_kv_heads")
    if (num_kv_heads is not None and map_heads is not None
            and int(map_heads) != num_kv_heads):
        logger.warning(
            "Bundled cluster map %s has num_kv_heads=%d but the model has %d; "
            "falling back to identity grouping.",
            path, int(map_heads), num_kv_heads)
        return False

    return True


def resolve_bundled_cluster_map(
    *,
    scorer: str,
    model_name: str,
    page_group_size: int,
    cluster_map_scope: str | None,
    num_kv_heads: int | None,
    tp_world_size: int,
) -> str | None:
    """Path of the bundled cluster map for this run, or ``None`` for identity
    grouping.

    Five things lose the technique and go through
    :func:`_fall_back_to_identity`: TP>1, an unknown scope, a scorer that ships
    no map, no bundled tree, no filename match, and a metadata mismatch. A
    sixth, ``cluster_map_scope is None``, is the ``uniform`` scope working as
    configured. ``cluster_map_scope`` is the budget scope's
    ``BudgetScope.cluster_map_scope``."""
    global identity_fallback_reason
    identity_fallback_reason = None

    if tp_world_size > 1:
        _fall_back_to_identity(
            f"the bundled maps are TP=1 layouts and tensor parallelism "
            f"(size {tp_world_size}) is in use")
        return None

    # Not a fallback: the ``uniform`` scope gives every entry the same count,
    # so there is no cross-entry comparison for a map to inform.
    if cluster_map_scope is None:
        return None

    suffix = _SCOPE_FILENAME_SUFFIX.get(cluster_map_scope)
    if suffix is None:
        _fall_back_to_identity(
            f"cluster map scope {cluster_map_scope!r} is not one of "
            f"{sorted(_SCOPE_FILENAME_SUFFIX)}")
        return None

    scorer_dir = _scorer_dir(scorer)
    if scorer_dir is None:
        _fall_back_to_identity(
            f"scorer {scorer!r} ships no cluster map (clustering by retention "
            "is meaningless for a recency or head-uniform scorer)")
        return None

    base_dir = _bundled_maps_base_dir()
    if base_dir is None:
        _fall_back_to_identity(
            "no bundled cluster map tree was found -- a wheel install carries "
            f"no tools/ directory; set {_CLUSTER_MAPS_DIR_ENV} to point at one")
        return None

    model_slug = _model_slug(model_name)
    map_dir = base_dir / scorer_dir / model_slug
    pattern = f"pg{page_group_size}_r*{suffix}.npz"
    matches = sorted(map_dir.glob(pattern))
    if suffix == "":
        # pg<N>_r*.npz also matches the _perlayer siblings; drop them so a
        # cross-layer (global) resolution never picks a per-layer map.
        matches = [m for m in matches if not m.stem.endswith("_perlayer")]

    if not matches:
        _fall_back_to_identity(
            f"no bundled map for scorer={scorer!r} model={model_name!r} "
            f"page_group_size={page_group_size} scope={cluster_map_scope!r} "
            f"(looked for {pattern} under {map_dir})")
        return None

    if len(matches) > 1:
        logger.warning(
            "Multiple cluster maps match %s under %s: %s; using %r (base_ratio "
            "is ignored, so any is valid).",
            pattern, map_dir, [m.name for m in matches], matches[0].name)

    path = str(matches[0])
    if not _validate_cluster_map_meta(
            path,
            expected_scope=cluster_map_scope,
            model_slug=model_slug,
            page_group_size=page_group_size,
            num_kv_heads=num_kv_heads):
        _fall_back_to_identity(
            f"the metadata in {path} does not describe this run (the "
            "mismatched field is named in the warning above)")
        return None

    logger.info("Auto-resolved head-group cluster map: %s", path)
    return path
