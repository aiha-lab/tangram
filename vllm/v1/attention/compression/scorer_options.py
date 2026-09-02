# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-scorer hyperparameters (axis 2), declared by the scorer.

A scorer declares its settings as ``OPTIONS`` and the engine carries them
through one generic channel (``CacheConfig.compression_scorer_options``), so a
new scorer or setting is a subclass plus one declaration -- no configuration,
CLI, entrypoint or factory signature changes. The declaration is the single
source of truth for the default, the accepted values and the help text, so
validation, the startup log and the error messages cannot drift from the code
that consumes them.

No torch import here on purpose: config validation and CLI parsing resolve
options through this module and must not pull the runtime scorers in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

#: Values accepted for a boolean option, spelled the way a CLI user would.
_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


@dataclass(frozen=True)
class ScorerOption:
    """One setting a scorer understands.

    Attributes:
        name: key as written on the command line, namespaced by the scorer.
        type: how the raw string is converted before the constructor sees it.
        default: THE default. Configuration must not carry a second copy.
        help: one sentence for ``--help`` and the startup log.
        choices: accepted values for a ``str`` option, else ``None``.
        requirement: ``(phrase, predicate)`` the converted value must satisfy,
            e.g. ``("a positive odd integer", lambda v: v > 0 and v % 2 == 1)``.
    """
    name: str
    type: type
    default: Any
    help: str
    choices: tuple[str, ...] | None = None
    requirement: tuple[str, Callable[[Any], bool]] | None = None


def parse_scorer_options(raw: str) -> dict[str, str]:
    """Parse ``"key=value,key=value"`` into a mapping of raw strings.

    Left as text because only the scorer's ``OPTIONS`` know each key's type;
    ``resolve_scorer_options`` converts. An empty string is an empty mapping, so
    ``--compression-scorer-options ""`` means "no options", not an error.
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
            value = True
        elif lowered in _FALSE:
            value = False
        else:
            raise ValueError(
                f"{where}: expected a boolean "
                f"({'/'.join(sorted(_TRUE))} or {'/'.join(sorted(_FALSE))}), "
                f"got {raw!r}.")
    elif option.type is str:
        if option.choices is not None and raw not in option.choices:
            raise ValueError(
                f"{where}: expected one of {option.choices}, got {raw!r}.")
        value = raw
    else:
        try:
            value = option.type(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{where}: expected {option.type.__name__}, "
                f"got {raw!r}.") from exc
    if option.requirement is not None:
        phrase, holds = option.requirement
        if not holds(value):
            raise ValueError(f"{where}: expected {phrase}, got {value!r}.")
    return value


def resolve_scorer_options(
    scorer_name: str,
    options_spec: Sequence[ScorerOption],
    given: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Apply ``given`` on top of the declared defaults, typed and validated.

    An unknown key raises -- it is a typo or a setting meant for another
    scorer, never something to ignore. The result is complete, one entry per
    declared option, so a constructor never re-states a default.
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
    """One line naming every setting in force, printed whether or not anything
    was overridden: behaviour depends on these, so a log should state them
    rather than leave a reader to infer defaults from the source.
    """
    if not options_spec:
        return f"scorer '{scorer_name}' has no options"
    settings = ", ".join(
        f"{option.name}={resolved[option.name]!r}" for option in options_spec)
    return f"scorer '{scorer_name}' options: {settings}"
