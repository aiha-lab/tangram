# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a gate checkpoint is loaded from, and in which order.

The gate repo is pinned to one revision so a run can be attributed to the
weights it used. Only two lookups may precede the Hub: an absolute path the
operator staged, and the Hub cache for that same pinned revision.
"""
import pytest

from vllm.v1.attention.compression import gate


@pytest.fixture
def no_hub(monkeypatch):
    """Neither Hub helper may run unless a test says so."""
    def unexpected(*args, **kwargs):
        raise AssertionError("Hub lookup should not have been reached")

    monkeypatch.setattr(gate, "_hf_cached_gate_path", unexpected)
    monkeypatch.setattr(gate, "_hf_download_gate_path", unexpected)


def test_staged_absolute_path_wins(tmp_path, no_hub):
    staged = tmp_path / "q4_dim16_sink16.pt"
    staged.write_bytes(b"")

    assert gate._download_or_local("any/model", str(staged)) == str(staged)


def test_hub_cache_before_download(monkeypatch):
    monkeypatch.setattr(gate, "_resolve_gate_filename",
                        lambda model, path: "m/q4_dim16_sink16.pt")
    monkeypatch.setattr(gate, "_hf_cached_gate_path", lambda resolved: "/cached")
    monkeypatch.setattr(
        gate, "_hf_download_gate_path",
        lambda resolved, path: pytest.fail("downloaded despite a cache hit"))

    assert gate._download_or_local("any/model", "fastkvzip") == "/cached"


def test_download_on_cache_miss(monkeypatch):
    monkeypatch.setattr(gate, "_resolve_gate_filename",
                        lambda model, path: "m/q4_dim16_sink16.pt")
    monkeypatch.setattr(gate, "_hf_cached_gate_path", lambda resolved: None)
    monkeypatch.setattr(gate, "_hf_download_gate_path",
                        lambda resolved, path: "/downloaded")

    assert gate._download_or_local("any/model", "fastkvzip") == "/downloaded"


def test_no_home_directory_lookup():
    """A home-directory scan would shadow the pinned revision (Q5)."""
    assert not hasattr(gate, "_local_gate_path")
