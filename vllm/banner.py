"""Tangram startup banner."""

import os
import sys

_BANNER = r"""
 _____
|_   _|_ _ _ __   __ _ _ __ __ _ _ __ ___
  | |/ _` | '_ \ / _` | '__/ _` | '_ ` _ \
  | | (_| | | | | (_| | | | (_| | | | | | |
  |_|\__,_|_| |_|\__, |_|  \__,_|_| |_| |_|
                 |___/
"""


def print_banner() -> None:
    """Print the Tangram banner once at engine startup.

    Skipped when stderr is not a terminal (CI, log files) or when
    ``TANGRAM_NO_BANNER`` is set.
    """
    if os.environ.get("TANGRAM_NO_BANNER"):
        return
    if not sys.stderr.isatty():
        return
    print(_BANNER, file=sys.stderr)
