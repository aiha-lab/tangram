# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-scorer hyperparameters (compression axis 2), declared by the scorer.

A scorer often has settings that only it understands — SnapKV's observation
window, KeyDiff's anchor formula, ExpectedAttention's covariance term. Giving
each of them its own configuration field, CLI flag and constructor parameter
makes adding a scorer a six-file change, which is exactly what the axis-2
registry exists to prevent. Instead a scorer declares its settings as
``OPTIONS``, and the engine carries them through one generic channel
(``CacheConfig.compression_scorer_options``):

    class KeyDiffScorer(QKScorer):
        OPTIONS = (ScorerOption("anchor", str, "unnormalized", "...",
                                choices=("unnormalized", "normalized")),)

    vllm serve ... --compression-scorer keydiff \\
                   --compression-scorer-options anchor=normalized

So a new scorer, or a new setting on an existing one, is a subclass plus one
declaration: no configuration, CLI, entrypoint or factory signature changes.
The declaration is also the single source of truth for the setting's DEFAULT,
its accepted values and its help text, so validation, the startup log and the
error messages cannot drift apart from the code that consumes it.

This module deliberately holds no torch import: configuration validation and
CLI parsing resolve options through it, and neither should pull the runtime
scorer modules into its import graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: Values accepted for a boolean option, spelled the way a CLI user would.
_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


@dataclass(frozen=True)
class ScorerOption:
    """One setting a scorer understands.

    Attributes:
        name: key as written on the command line (namespaced by the scorer, so
            short names like ``window`` are unambiguous).
        type: ``str`` / ``int`` / ``float`` / ``bool`` — how the raw string is
            converted before it reaches the scorer's constructor.
        default: value used when the setting is not given. This is THE default;
            configuration must not carry a second copy of it.
        help: one sentence for ``--help`` and the startup log.
        choices: accepted values for a ``str`` option, or ``None`` for free form.
    """
    name: str
    type: type
    default: Any
    help: str
    choices: tuple[str, ...] | None = None


def parse_scorer_options(raw: str) -> dict[str, str]:
    """Parse ``"key=value,key=value"`` into a mapping of raw strings.

    Values are left as text here because only the scorer's ``OPTIONS`` know
    each key's type; ``resolve_scorer_options`` does the conversion. An empty
    string is an empty mapping so ``--compression-scorer-options ""`` means
    "no options" rather than an error.
    """
    options: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"scorer option {item!r} must be written key=value "
                "(comma-separated for several).")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"scorer option {item!r} has an empty key; expected key=value.")
        if key in options:
            raise ValueError(
                f"scorer option {key!r} given more than once.")
        options[key] = value.strip()
    return options


def _convert(option: ScorerOption, raw: str, scorer_name: str) -> Any:
    """Turn one raw string into the option's declared type, or explain why not."""
    where = f"scorer option {scorer_name}.{option.name}"
    if option.type is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(
            f"{where}: expected a boolean "
            f"({'/'.join(sorted(_TRUE))} or {'/'.join(sorted(_FALSE))}), "
            f"got {raw!r}.")
    if option.type is str:
        if option.choices is not None and raw not in option.choices:
            raise ValueError(
                f"{where}: expected one of {option.choices}, got {raw!r}.")
        return raw
    try:
        return option.type(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{where}: expected {option.type.__name__}, got {raw!r}.") from exc


def resolve_scorer_options(
    scorer_name: str,
    options_spec: Sequence[ScorerOption],
    given: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Apply ``given`` on top of the declared defaults, typed and validated.

    Every key must be one this scorer declares: an unknown key is a mistake the
    user wants to hear about (a typo, or a setting meant for a different
    scorer), never something to ignore. The returned mapping is complete — one
    entry per declared option — so the scorer's constructor receives every
    setting explicitly and never has to re-state a default.
    """
    declared = {option.name: option for option in options_spec}
    resolved: dict[str, Any] = {
        option.name: option.default
        for option in options_spec
    }
    for key, raw in (given or {}).items():
        option = declared.get(key)
        if option is None:
            known = tuple(declared) or ("<none>", )
            raise ValueError(
                f"unknown scorer option {key!r} for compression_scorer "
                f"{scorer_name!r}; it accepts {known}.")
        resolved[key] = _convert(option, raw, scorer_name)
    return resolved


def describe_scorer_options(
    scorer_name: str,
    options_spec: Sequence[ScorerOption],
    resolved: Mapping[str, Any],
) -> str:
    """One line naming every setting in force, for the startup log.

    Printed whether or not anything was overridden: a scorer's behaviour
    depends on these, so a run's log should state them rather than leave the
    reader to infer defaults from the source.
    """
    if not options_spec:
        return f"scorer '{scorer_name}' has no options"
    settings = ", ".join(
        f"{option.name}={resolved[option.name]!r}" for option in options_spec)
    return f"scorer '{scorer_name}' options: {settings}"
