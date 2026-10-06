"""Run-scoped material snapshots; all workspaces and mutations are temporary."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

import run_source_store as store_module
from run_source_store import (
    RunSourceStore,
    RunSourceStoreCorruptionError,
    RunSourceStoreError,
    RunSourceStoreValidationError,
)
from run_workspace import ResearchRunWorkspace


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    unused_root = tmp_path / "unused-environment-root"
    monkeypatch.setenv("RESEARCH_RUN_ROOT", str(unused_root))
    workspace = ResearchRunWorkspace.create("Example", "company", root_dir=tmp_path / "runs")
    assert not workspace.root.is_relative_to(Path(__file__).resolve().parents[1])
    yield workspace
    assert not unused_root.exists()


@pytest.fixture
def store(workspace):
    return RunSourceStore(workspace)


@pytest.fixture
def material():
    return {
        "source_ref": "  mock://example/report  ",
        "content": "Growth \u589e\u957f\r\nRevenue was 100.\n\nMargin was 20%.\n",
        "title": "Mock report", "source_type": "mock", "locator": "mock://example/report",
        "publisher": "Example publisher", "published_date": "2026-10-06",
        "content_type": "text/html", "primary_or_secondary": "Primary", "independence_group": "example",
    }


def source_paths(workspace, key):
    return (
        workspace.sources_raw_dir / f"{key}.raw",
        workspace.sources_normalized_dir / f"{key}.txt",
        workspace.sources_normalized_dir / f"{key}.json",
    )


def snapshot(workspace):
    return {path.relative_to(workspace.root): path.read_bytes()
            for path in (workspace.root / "sources").rglob("*") if path.is_file()}


@pytest.mark.parametrize("raw_content", [None, "<html>\r\nVisible \u6570\u636e\r\n</html>"], ids=["normalized-only", "raw-and-normalized"])
def test_put_get_roundtrip_metadata_and_integrity(store, workspace, material, raw_content):
    if raw_content is not None:
        material["raw_content"] = raw_content
    # Unknown fields, including credentials, never enter a snapshot.
    material.update(api_key="ignored-secret", headers={"Authorization": "ignored-secret"},
                    credentials="ignored-secret", SEC_USER_AGENT="ignored-secret", extra="ignored-secret")
    key = store.put(material)
    assert key == sha256(material["source_ref"].encode("utf-8")).hexdigest()
    assert key != sha256(material["source_ref"].strip().encode("utf-8")).hexdigest()
    raw, normalized, metadata_path = source_paths(workspace, key)
    assert normalized.read_bytes() == material["content"].encode("utf-8")
    metadata = json.loads(metadata_path.read_bytes())
    assert set(metadata) == {
        "source_ref", "title", "source_type", "locator", "publisher", "published_date", "content_type",
        "primary_or_secondary", "independence_group", "raw_file", "normalized_file",
        "raw_sha256", "normalized_sha256", "saved_at",
    }
    for field in ("source_ref", "title", "source_type", "locator", "publisher", "published_date",
                  "content_type", "primary_or_secondary", "independence_group"):
        assert metadata[field] == material[field]
    assert metadata["normalized_file"] == f"sources/normalized/{key}.txt"
    assert metadata["normalized_sha256"] == sha256(normalized.read_bytes()).hexdigest()
    assert datetime.fromisoformat(metadata["saved_at"]).tzinfo == timezone.utc
    if raw_content is None:
        assert not raw.exists() and metadata["raw_file"] is None and metadata["raw_sha256"] is None
    else:
        assert raw.read_bytes() == raw_content.encode("utf-8")
        assert metadata["raw_file"] == f"sources/raw/{key}.raw"
        assert metadata["raw_sha256"] == sha256(raw.read_bytes()).hexdigest()
    loaded = store.get(material["source_ref"])
    assert loaded == {**metadata, "content": material["content"], **({"raw_content": raw_content} if raw_content is not None else {})}
    assert store.contains(material["source_ref"])
    reopened = ResearchRunWorkspace.open(workspace.root, root_dir=workspace.root.parent)
    assert RunSourceStore(reopened).get(material["source_ref"]) == loaded
    assert "ignored-secret" not in metadata_path.read_text(encoding="utf-8")


def test_missing_distinct_sources_and_two_runs_are_isolated(store, workspace, tmp_path):
    assert store.get("missing") is None and not store.contains("missing")
    assert snapshot(workspace) == {}
    key_a = store.put({"source_ref": "first", "content": "Run A"})
    key_b = store.put({"source_ref": "second", "content": "Another source"})
    assert key_a != key_b
    assert store.get("first")["content"] == "Run A"
    assert store.get("second")["content"] == "Another source"
    second = ResearchRunWorkspace.create("Example", "company", root_dir=tmp_path / "runs")
    other = RunSourceStore(second)
    assert other.get("first") is None and not other.contains("first")
    assert other.put({"source_ref": "first", "content": "Run B"}) == key_a
    assert other.get("first")["content"] == "Run B"
    assert store.get("first")["content"] == "Run A"


def test_identical_put_keeps_all_files_and_saved_at(store, workspace, material, monkeypatch):
    material["raw_content"] = ""
    key = store.put(material)
    raw, _, _ = source_paths(workspace, key)
    assert raw.is_file() and raw.read_bytes() == b""
    before = snapshot(workspace)
    monkeypatch.setattr(store_module, "_now_utc", lambda: datetime(2040, 1, 1, tzinfo=timezone.utc))

    def unexpected_replace(*args):
        pytest.fail("An identical snapshot must not replace its files")

    monkeypatch.setattr(store_module.os, "replace", unexpected_replace)
    assert store.put(dict(material)) == key
    assert snapshot(workspace) == before
    assert store.get(material["source_ref"])["raw_content"] == ""


def test_changed_put_replaces_snapshot_and_removes_absent_raw(store, workspace, material):
    material["raw_content"] = "Old raw"
    key = store.put(material)
    material.update(content="New normalized", raw_content="New raw", title="New title")
    assert store.put(material) == key
    loaded = store.get(material["source_ref"])
    assert (loaded["content"], loaded["raw_content"], loaded["title"]) == ("New normalized", "New raw", "New title")
    material.pop("raw_content")
    material["content"] = "Normalized-only replacement"
    assert store.put(material) == key
    raw, normalized, metadata_path = source_paths(workspace, key)
    loaded = store.get(material["source_ref"])
    assert loaded["content"] == material["content"] and "raw_content" not in loaded
    assert loaded["raw_file"] is None and loaded["raw_sha256"] is None and not raw.exists()
    assert set(snapshot(workspace)) == {normalized.relative_to(workspace.root), metadata_path.relative_to(workspace.root)}


@pytest.mark.parametrize("part", ["normalized", "raw"])
def test_content_hash_mismatch_is_rejected_without_repair(store, workspace, material, part):
    material["raw_content"] = "Original raw"
    key = store.put(material)
    raw, normalized, _ = source_paths(workspace, key)
    path = raw if part == "raw" else normalized
    path.write_bytes(b"Tampered content")
    before = snapshot(workspace)
    for lookup in (store.get, store.contains):
        with pytest.raises(RunSourceStoreCorruptionError, match="SHA-256 mismatch"):
            lookup(material["source_ref"])
    assert snapshot(workspace) == before


@pytest.mark.parametrize("damage", ["json", "schema", "path", "identity", "missing-content"])
def test_invalid_or_incomplete_snapshot_is_rejected(store, workspace, material, damage):
    key = store.put(material)
    _, normalized, metadata_path = source_paths(workspace, key)
    if damage == "json":
        metadata_path.write_bytes(b"{broken")
    elif damage == "missing-content":
        normalized.unlink()
    else:
        metadata = json.loads(metadata_path.read_bytes())
        if damage == "schema":
            metadata["title"] = ["Invalid type"]
        elif damage == "path":
            metadata["normalized_file"] = "../outside.txt"
        else:
            metadata["source_ref"] = "Another source"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    before = snapshot(workspace)
    with pytest.raises(RunSourceStoreCorruptionError):
        store.get(material["source_ref"])
    assert snapshot(workspace) == before


@pytest.mark.parametrize("material", [
    {"source_ref": " \t", "content": "Text"},
    {"source_ref": "source"},
    {"source_ref": "source", "content": 123},
    {"source_ref": "source", "content": "Text", "title": {}},
    {"source_ref": "source", "content": "Text", "raw_content": b"Bytes"},
    [],
], ids=["blank-ref", "missing-content", "invalid-content", "invalid-metadata", "invalid-raw", "not-dict"])
def test_invalid_input_is_rejected_before_writing(store, workspace, material):
    with pytest.raises(RunSourceStoreValidationError):
        store.put(material)
    assert snapshot(workspace) == {}


def test_source_ref_cannot_be_used_for_path_traversal(store, workspace):
    source_ref = "../../outside/company\\secret?name=report"
    key = store.put({"source_ref": source_ref, "content": "Text"})
    assert key == sha256(source_ref.encode("utf-8")).hexdigest()
    _, normalized, metadata_path = source_paths(workspace, key)
    assert set(snapshot(workspace)) == {normalized.relative_to(workspace.root), metadata_path.relative_to(workspace.root)}
    assert store.get(source_ref)["content"] == "Text"


@pytest.mark.parametrize("entry", ["raw-leaf", "normalized-leaf", "metadata-leaf", "sources-parent"])
def test_symlink_escape_rejected_for_put_get_and_contains(store, workspace, material, tmp_path, entry):
    material["raw_content"] = "Original raw"
    key = store.put(material)
    outside = tmp_path / "outside"
    outside.mkdir()
    if entry == "sources-parent":
        sources = workspace.root / "sources"
        saved_sources = workspace.root / "saved-sources"
        sources.rename(saved_sources)
        # Even when both children lead back inside the run, the ancestor escapes.
        (outside / "raw").symlink_to(saved_sources / "raw", target_is_directory=True)
        (outside / "normalized").symlink_to(saved_sources / "normalized", target_is_directory=True)
        sources.symlink_to(outside, target_is_directory=True)
    else:
        path = source_paths(workspace, key)[{"raw-leaf": 0, "normalized-leaf": 1, "metadata-leaf": 2}[entry]]
        target = outside / path.name
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    before = {path: path.read_bytes() for path in outside.iterdir() if path.is_file()}
    for operation in (lambda: store.put(material), lambda: store.get(material["source_ref"]),
                      lambda: store.contains(material["source_ref"])):
        with pytest.raises(RunSourceStoreValidationError, match="boundary|symlink"):
            operation()
    assert {path: path.read_bytes() for path in outside.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("failure_at", ["normalized", "metadata"])
def test_atomic_replace_failure_is_explicit_and_cleans_temporaries(store, workspace, monkeypatch, failure_at):
    key = store.put({"source_ref": "source", "content": "Original"})
    _, normalized, metadata_path = source_paths(workspace, key)
    before = snapshot(workspace)
    replace = store_module.os.replace

    def fail_replace(temporary, destination):
        target = metadata_path if failure_at == "metadata" else normalized
        assert Path(temporary).parent == Path(destination).parent
        if Path(destination) == target:
            raise OSError("simulated replace failure")
        replace(temporary, destination)

    monkeypatch.setattr(store_module.os, "replace", fail_replace)
    with pytest.raises(RunSourceStoreError, match="simulated replace failure") as error:
        store.put({"source_ref": "source", "content": "Changed"})
    assert isinstance(error.value.__cause__, OSError)
    assert set(snapshot(workspace)) == set(before)  # No staging files remain.
    assert metadata_path.read_bytes() == before[metadata_path.relative_to(workspace.root)]
    if failure_at == "normalized":
        assert snapshot(workspace) == before
        assert store.get("source")["content"] == "Original"
    else:
        assert normalized.read_bytes() == b"Changed"
        with pytest.raises(RunSourceStoreCorruptionError, match="SHA-256 mismatch"):
            store.get("source")
