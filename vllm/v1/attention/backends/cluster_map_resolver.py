# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-resolution of bundled head-group cluster maps.

Finds the ``.npz`` map shipped under
``tools/head_group_clustering/cluster_maps/<scorer-dir>/<model-slug>/`` that
matches the running model, scorer, ``page_group_size``, and selection level, so
the default config needs no explicit ``--head-group-cluster-map``. This only
*locates and identity-checks* a file; ``ragged_layout.load_cluster_map``
stays the authoritative loader/validator of the array contents. Run once at
config finalization (``CacheConfig.resolve_head_group_cluster_map``) and frozen
onto the config, so every consumer reads the same path.

Filename convention (see the cluster_maps README):
``<scorer-dir>/<model-slug>/pg<page_group_size>_r<ratio>[_perlayer].npz``.
The ratio is informational (the runtime ignores ``base_ratio``), so it is
matched with a glob.
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
            "Bundled cluster map %s has cluster_scope=%r but the selection "
            "level needs %r; falling back to identity grouping.",
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
    grouping. Returns ``None`` (logging why) when a map cannot be confidently
    resolved: TP>1 (maps are TP=1 layouts), a level with no map
    (``cluster_map_scope is None``), a scorer/tree with no map, no filename
    match, or a metadata mismatch. ``cluster_map_scope`` is the level's
    ``SelectionLevel.cluster_map_scope``."""
    if tp_world_size > 1:
        logger.info(
            "Tensor parallelism (size %d) in use; cluster maps are TP=1 "
            "layouts, so using identity grouping.", tp_world_size)
        return None

    if cluster_map_scope is None:  # level uses no cluster map (e.g. uniform)
        return None

    suffix = _SCOPE_FILENAME_SUFFIX.get(cluster_map_scope)
    if suffix is None:
        logger.warning(
            "Unknown cluster map scope %r; using identity grouping.",
            cluster_map_scope)
        return None

    scorer_dir = _scorer_dir(scorer)
    if scorer_dir is None:
        logger.info(
            "Scorer %r ships no cluster map; using identity grouping.", scorer)
        return None

    base_dir = _bundled_maps_base_dir()
    if base_dir is None:
        logger.info(
            "No bundled cluster map tree found (set %s to override); using "
            "identity grouping.", _CLUSTER_MAPS_DIR_ENV)
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
        logger.warning(
            "No bundled cluster map for scorer=%r model=%r page_group_size=%d "
            "scope=%r (looked for %s under %s); using identity grouping.",
            scorer, model_name, page_group_size, cluster_map_scope,
            pattern, map_dir)
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
        return None

    logger.info("Auto-resolved head-group cluster map: %s", path)
    return path
