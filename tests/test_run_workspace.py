"""Workspace creation and data boundaries; every filesystem fixture is temporary."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

import run_workspace as workspace_module
from run_workspace import (
    ResearchRunWorkspace,
    RunWorkspaceConfigurationError,
    RunWorkspaceError,
    RunWorkspaceValidationError,
)


NOW = datetime(2026, 10, 6, 15, 30, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_configuration_and_clock(monkeypatch):
    # A developer's real run root must never be used by these tests.
    monkeypatch.delenv("RESEARCH_RUN_ROOT", raising=False)
    monkeypatch.setattr(workspace_module, "_now_utc", lambda: NOW)


@pytest.fixture
def root(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def workspace(root):
    return ResearchRunWorkspace.create("PL", "company", root_dir=root)


def test_explicit_root_layout_manifest_and_reopen(workspace, root):
    assert workspace.root.parent == root.resolve()
    assert not workspace.root.is_relative_to(REPO)
    assert re.fullmatch(r"20261006_153000_PL_[0-9a-f]{8}", workspace.root.name)
    paths = {
        "sources/raw": workspace.sources_raw_dir,
        "sources/normalized": workspace.sources_normalized_dir,
        "objects": workspace.objects_dir,
        "intermediate": workspace.intermediate_dir,
        "logs": workspace.logs_dir,
        "outputs": workspace.outputs_dir,
    }
    for relative, path in paths.items():
        assert path == workspace.root / relative
        assert path.is_dir() and list(path.iterdir()) == []
    assert {str(path.relative_to(workspace.root)) for path in workspace.root.rglob("*")} == {
        "manifest.json", "sources", *paths,
    }
    manifest = json.loads((workspace.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == workspace.manifest == {
        "run_id": workspace.root.name, "subject": "PL", "research_type": "company",
        "created_at": NOW.isoformat(), "workspace_path": str(workspace.root), "status": "active",
    }
    assert datetime.fromisoformat(manifest["created_at"]).tzinfo == timezone.utc
    reopened = ResearchRunWorkspace.open(workspace.root, root_dir=root)
    assert reopened.root == workspace.root and reopened.manifest == manifest
    returned_manifest = reopened.manifest
    returned_manifest["subject"] = "Changed by caller"
    assert reopened.manifest == manifest


@pytest.mark.parametrize("explicit", [False, True], ids=["environment-root", "explicit-overrides-environment"])
def test_root_configuration_precedence(tmp_path, monkeypatch, explicit):
    env_root, explicit_root = tmp_path / "env-runs", tmp_path / "explicit-runs"
    monkeypatch.setenv("RESEARCH_RUN_ROOT", str(env_root))
    chosen_root = explicit_root if explicit else env_root
    workspace = ResearchRunWorkspace.create("EXM", "company", root_dir=explicit_root if explicit else None)
    assert workspace.root.parent == chosen_root.resolve()
    reopened = ResearchRunWorkspace.open(workspace.root, root_dir=explicit_root if explicit else None)
    assert reopened.manifest == workspace.manifest
    if explicit:
        assert not env_root.exists()


def test_missing_root_does_not_default_to_repo(tmp_path):
    for operation in (
        lambda: ResearchRunWorkspace.create("PL", "company"),
        lambda: ResearchRunWorkspace.open(tmp_path / "existing-run"),
    ):
        with pytest.raises(RunWorkspaceConfigurationError, match="RESEARCH_RUN_ROOT"):
            operation()
    assert list(tmp_path.iterdir()) == []


def test_two_creations_are_independent_and_collision_never_overwrites(root, monkeypatch):
    first = ResearchRunWorkspace.create("PL", "company", root_dir=root)
    marker = first.outputs_dir / "marker.txt"
    marker.write_text("First run", encoding="utf-8")
    original_manifest = (first.root / "manifest.json").read_bytes()
    second = ResearchRunWorkspace.create("PL", "company", root_dir=root)
    assert first.root != second.root
    assert second.outputs_dir != first.outputs_dir and not (second.outputs_dir / marker.name).exists()
    suffix = first.root.name.rsplit("_", 1)[1]
    monkeypatch.setattr(workspace_module, "uuid4", lambda: SimpleNamespace(hex=suffix + "0" * 24))
    with pytest.raises(RunWorkspaceError) as error:
        ResearchRunWorkspace.create("PL", "company", root_dir=root)
    assert isinstance(error.value.__cause__, FileExistsError)
    assert marker.read_text(encoding="utf-8") == "First run"
    assert (first.root / "manifest.json").read_bytes() == original_manifest
    assert set(root.iterdir()) == {first.root, second.root}


@pytest.mark.parametrize(("subject", "sanitized"), [
    ("Planet Labs PBC", "Planet_Labs_PBC"),
    ("Optical / Interconnect", "Optical_Interconnect"),
    ("../Optical \\ Interconnect...___", "Optical_Interconnect"),
])
def test_subject_is_sanitized_for_directory_name(root, subject, sanitized):
    workspace = ResearchRunWorkspace.create(subject, "industry", root_dir=root)
    assert re.fullmatch(rf"20261006_153000_{sanitized}_[0-9a-f]{{8}}", workspace.root.name)
    assert ".." not in workspace.root.name and workspace.root.parent == root.resolve()
    assert workspace.manifest["subject"] == subject


@pytest.mark.parametrize(("field", "value"), [
    ("subject", " \t"), ("subject", "../.."), ("research_type", ""),
])
def test_blank_or_unsanitizable_inputs_rejected_before_creating_directories(root, field, value):
    fields = {"subject": "PL", "research_type": "company", field: value}
    with pytest.raises(RunWorkspaceValidationError, match=field):
        ResearchRunWorkspace.create(**fields, root_dir=root)
    assert not root.exists()


@pytest.mark.parametrize("damage", ["missing", "json", "required-field", "run-id", "workspace-path", "non-utc"])
def test_open_rejects_damaged_manifest_without_repair(workspace, root, tmp_path, damage):
    path = workspace.root / "manifest.json"
    if damage == "missing":
        path.unlink()
    elif damage == "json":
        path.write_text("{broken", encoding="utf-8")
    else:
        manifest = workspace.manifest
        if damage == "required-field":
            manifest["research_type"] = None
        elif damage == "run-id":
            manifest["run_id"] = "wrong-run"
        elif damage == "workspace-path":
            manifest["workspace_path"] = str(tmp_path / "outside-root")
        else:
            manifest["created_at"] = "2026-10-06T15:30:00+01:00"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(RunWorkspaceValidationError):
        ResearchRunWorkspace.open(workspace.root, root_dir=root)
    assert (path.read_bytes() if path.exists() else None) == before


def test_open_does_not_rebuild_missing_directory(workspace, root):
    workspace.objects_dir.rmdir()
    with pytest.raises(RunWorkspaceValidationError, match="objects"):
        ResearchRunWorkspace.open(workspace.root, root_dir=root)
    assert not workspace.objects_dir.exists()


@pytest.mark.parametrize("escape", ["outside-root", "parent-component"])
def test_open_rejects_workspace_escape_and_lexical_traversal(workspace, root, tmp_path, escape):
    if escape == "outside-root":
        path = ResearchRunWorkspace.create("OTHER", "company", root_dir=tmp_path / "other-runs").root
    else:
        path = workspace.root / ".." / workspace.root.name
    with pytest.raises(RunWorkspaceValidationError):
        ResearchRunWorkspace.open(path, root_dir=root)


@pytest.mark.parametrize("entry", ["run", "manifest", "raw", "sources-parent"])
def test_open_rejects_symlink_escapes(workspace, root, tmp_path, entry):
    outside = tmp_path / "outside"
    outside.mkdir()
    path = workspace.root
    if entry == "run":
        path = root / "linked-run"
        path.symlink_to(outside, target_is_directory=True)
    elif entry == "manifest":
        manifest_path = workspace.root / "manifest.json"
        outside_manifest = outside / "manifest.json"
        outside_manifest.write_text(json.dumps(workspace.manifest), encoding="utf-8")
        manifest_path.unlink()
        manifest_path.symlink_to(outside_manifest)
    elif entry == "raw":
        # Even another run inside the configured root is outside this run's boundary.
        other = ResearchRunWorkspace.create("OTHER", "company", root_dir=root)
        workspace.sources_raw_dir.rmdir()
        workspace.sources_raw_dir.symlink_to(other.sources_raw_dir, target_is_directory=True)
    else:
        # Children resolve back inside this run, but their sources ancestor escapes.
        sources = workspace.root / "sources"
        saved_sources = workspace.root / "saved-sources"
        sources.rename(saved_sources)
        (outside / "raw").symlink_to(saved_sources / "raw", target_is_directory=True)
        (outside / "normalized").symlink_to(saved_sources / "normalized", target_is_directory=True)
        sources.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RunWorkspaceValidationError, match="boundary"):
        ResearchRunWorkspace.open(path, root_dir=root)


def test_filesystem_failure_is_explicit_and_existing_data_is_untouched(tmp_path):
    blocked_root = tmp_path / "root-is-a-file"
    blocked_root.write_text("Existing data", encoding="utf-8")
    with pytest.raises(RunWorkspaceError) as error:
        ResearchRunWorkspace.create("PL", "company", root_dir=blocked_root)
    assert isinstance(error.value.__cause__, OSError)
    assert blocked_root.read_text(encoding="utf-8") == "Existing data"
