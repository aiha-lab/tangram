# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned upstream commit must still describe this tree's compiled surface.

Under ``VLLM_USE_PRECOMPILED=1`` the compiled extensions come from the wheel
built at the commit in ``.vllm_base_commit`` rather than from a local build.
That wheel is correct precisely while nothing Python cannot express has changed
since the commit, which is what lets the pin exist at all: Tangram's work is
Python and Triton, so the compiled surface is still upstream's.

Rebasing onto a newer upstream moves that surface, so it must move the pin too.
This test is that check. It needs no network, and reports which paths diverged.
"""

import subprocess
from pathlib import Path

import pytest

# Everything the prebuilt wheel bakes in: the sources compiled into it, the
# build system that shapes them, and the torch version they link against.
COMPILED_SURFACE = (
    "csrc",
    "cmake",
    "CMakeLists.txt",
    "requirements/build.txt",
)

PIN_FILE = ".vllm_base_commit"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


@pytest.fixture(scope="module")
def pinned_commit() -> str:
    pin = Path(git("rev-parse", "--show-toplevel")) / PIN_FILE
    if not pin.is_file():
        pytest.skip(f"no {PIN_FILE} in this tree")
    commit = pin.read_text().split(maxsplit=1)[0]
    if subprocess.run(["git", "cat-file", "-e", commit]).returncode != 0:
        # A shallow clone legitimately lacks the commit; a complete one that
        # lacks it means the pin names nothing real, which is not a skip.
        if git("rev-parse", "--is-shallow-repository") == "true":
            pytest.skip(f"{commit} is outside this shallow clone")
        pytest.fail(f"{PIN_FILE} names {commit}, which is not a commit")
    return commit


def test_pin_is_an_ancestor_of_head(pinned_commit):
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", pinned_commit, "HEAD"]
        ).returncode
        == 0
    ), (
        f"{PIN_FILE} names {pinned_commit[:12]}, which is not in HEAD's "
        "history, so its wheel was not built from this tree's upstream."
    )


def test_compiled_surface_is_unchanged_since_the_pin(pinned_commit):
    diverged = git(
        "diff", "--name-only", pinned_commit, "HEAD", "--", *COMPILED_SURFACE
    )
    assert not diverged, (
        f"the compiled surface moved since {pinned_commit[:12]}, so its "
        f"prebuilt wheel no longer matches this tree:\n{diverged}\n"
        f"Update {PIN_FILE} to the upstream commit this tree now sits on."
    )
