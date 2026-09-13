"""Tests for portable external-resource path resolution."""

from projects.amplitude_symbol.paths import (
    DATA_PATH_ENV,
    MANIFEST_PATH_ENV,
    UPSTREAM_ROOT_ENV,
    resolve_data_path,
    resolve_manifest_path,
    resolve_upstream_root,
)


def test_explicit_paths_take_priority_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_PATH_ENV, str(tmp_path / "env-data"))
    monkeypatch.setenv(UPSTREAM_ROOT_ENV, str(tmp_path / "env-upstream"))
    monkeypatch.setenv(MANIFEST_PATH_ENV, str(tmp_path / "env-manifest"))

    assert resolve_data_path(tmp_path / "explicit-data") == tmp_path / "explicit-data"
    assert resolve_upstream_root(tmp_path / "explicit-upstream") == tmp_path / "explicit-upstream"
    assert resolve_manifest_path(tmp_path / "explicit-manifest") == tmp_path / "explicit-manifest"


def test_environment_paths_are_used_when_explicit_values_are_absent(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    upstream_root = tmp_path / "upstream"
    manifest_path = tmp_path / "MANIFEST.json"
    monkeypatch.setenv(DATA_PATH_ENV, str(data_path))
    monkeypatch.setenv(UPSTREAM_ROOT_ENV, str(upstream_root))
    monkeypatch.setenv(MANIFEST_PATH_ENV, str(manifest_path))

    assert resolve_data_path() == data_path
    assert resolve_upstream_root() == upstream_root
    assert resolve_manifest_path() == manifest_path
