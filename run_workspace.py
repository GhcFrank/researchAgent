"""One research run's filesystem workspace, independent of tools and Storage.

No default root, cleanup, or integrations. Both factories require an explicit
root_dir or RESEARCH_RUN_ROOT. Paths are resolved before containment checks;
this is not a sandbox against concurrent filesystem or symlink replacement.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4


class RunWorkspaceError(Exception):
    """Base error, including filesystem failures creating or opening a run."""


class RunWorkspaceConfigurationError(RunWorkspaceError):
    """No usable run root was explicitly configured."""


class RunWorkspaceValidationError(RunWorkspaceError):
    """Inputs, workspace structure, manifest, or containment are invalid."""


_DIRECTORIES = ("sources", "sources/raw", "sources/normalized", "objects", "intermediate", "logs", "outputs")
_MANIFEST_FIELDS = ("run_id", "subject", "research_type", "created_at", "workspace_path")


def _nonblank(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunWorkspaceValidationError(f"{field} must be a non-blank string")
    return value.strip()


def _sanitize_subject(subject: str) -> str:
    subject = _nonblank(subject, "subject")
    sanitized = re.sub(r"[^\w]+", "_", subject)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        raise RunWorkspaceValidationError("subject must contain letters or digits after sanitization")
    return sanitized


def _configured_root(root_dir: str | Path | None) -> Path:
    configured = root_dir if root_dir is not None else os.getenv("RESEARCH_RUN_ROOT")
    if not isinstance(configured, (str, Path)) or (isinstance(configured, str) and not configured.strip()):
        raise RunWorkspaceConfigurationError("Set RESEARCH_RUN_ROOT or pass an explicit root_dir")
    try:
        return Path(configured).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunWorkspaceConfigurationError(f"Cannot resolve the configured run root: {exc}") from exc


def _contained_path(path: str | Path, boundary: Path) -> Path:
    if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path.strip()):
        raise RunWorkspaceValidationError("workspace path must be a non-blank path")
    try:
        candidate = Path(path).expanduser()
        if ".." in candidate.parts:
            raise RunWorkspaceValidationError("Path traversal is not allowed in workspace paths")
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunWorkspaceValidationError(f"Cannot resolve workspace path {path}: {exc}") from exc
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise RunWorkspaceValidationError(f"Workspace path {candidate} escapes boundary {boundary}")
    return resolved


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _validate_manifest(manifest, workspace_path: Path, root_dir: Path) -> None:
    if not isinstance(manifest, dict):
        raise RunWorkspaceValidationError("manifest.json must contain an object")
    for field in _MANIFEST_FIELDS:
        _nonblank(manifest.get(field), f"manifest.{field}")
    _sanitize_subject(manifest["subject"])
    if manifest["run_id"] != workspace_path.name:
        raise RunWorkspaceValidationError("manifest.run_id does not match the workspace directory")
    recorded_path = Path(manifest["workspace_path"])
    if not recorded_path.is_absolute() or _contained_path(recorded_path, root_dir) != workspace_path:
        raise RunWorkspaceValidationError("manifest.workspace_path does not match the actual workspace")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"])
    except ValueError as exc:
        raise RunWorkspaceValidationError("manifest.created_at must be an ISO timestamp") from exc
    if created_at.utcoffset() != timedelta(0):
        raise RunWorkspaceValidationError("manifest.created_at must have a UTC timezone")
    if "status" in manifest:
        _nonblank(manifest["status"], "manifest.status")


@dataclass(frozen=True)
class ResearchRunWorkspace:
    """Use create/open factories; public properties expose one validated run.

    root is the run directory, not its configured parent. manifest returns a
    defensive copy. Creating a run never merges an existing directory; creation
    failure may leave a partial run, which open rejects without rebuilding it.
    """

    _root: Path
    _manifest: dict

    @classmethod
    def create(
        cls, subject: str, research_type: str, root_dir: str | Path | None = None,
    ) -> "ResearchRunWorkspace":
        configured_root = _configured_root(root_dir)
        subject = _nonblank(subject, "subject")
        sanitized_subject = _sanitize_subject(subject)
        research_type = _nonblank(research_type, "research_type")
        created_at = _now_utc()
        run_id = f"{created_at:%Y%m%d_%H%M%S}_{sanitized_subject}_{uuid4().hex[:8]}"
        workspace_path = _contained_path(configured_root / run_id, configured_root)
        manifest = {
            "run_id": run_id,
            "subject": subject,
            "research_type": research_type,
            "created_at": created_at.isoformat(),
            "workspace_path": str(workspace_path),
            "status": "active",
        }
        try:
            configured_root.mkdir(parents=True, exist_ok=True)
            workspace_path.mkdir(exist_ok=False)
            for directory in _DIRECTORIES:
                target = _contained_path(workspace_path / directory, workspace_path)
                target.mkdir(exist_ok=False)
            with (workspace_path / "manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except OSError as exc:
            raise RunWorkspaceError(f"Cannot create run workspace {workspace_path}: {exc}") from exc
        return cls.open(workspace_path, root_dir=configured_root)

    @classmethod
    def open(
        cls, workspace_path: str | Path, root_dir: str | Path | None = None,
    ) -> "ResearchRunWorkspace":
        """Validate an existing run under an independently configured root."""
        configured_root = _configured_root(root_dir)
        workspace_path = _contained_path(workspace_path, configured_root)
        try:
            if not workspace_path.is_dir():
                raise RunWorkspaceValidationError(f"Workspace directory does not exist: {workspace_path}")
            manifest_path = _contained_path(workspace_path / "manifest.json", workspace_path)
            if not manifest_path.is_file():
                raise RunWorkspaceValidationError("Workspace manifest.json is missing or is not a file")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RunWorkspaceValidationError("Workspace manifest.json is damaged") from exc
            _validate_manifest(manifest, workspace_path, configured_root)
            for directory in _DIRECTORIES:
                target = _contained_path(workspace_path / directory, workspace_path)
                if not target.is_dir():
                    raise RunWorkspaceValidationError(f"Required workspace directory is missing: {directory}")
        except OSError as exc:
            raise RunWorkspaceError(f"Cannot open run workspace {workspace_path}: {exc}") from exc
        return cls(_root=workspace_path, _manifest=manifest)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def sources_raw_dir(self) -> Path:
        return self._root / "sources" / "raw"

    @property
    def sources_normalized_dir(self) -> Path:
        return self._root / "sources" / "normalized"

    @property
    def objects_dir(self) -> Path:
        return self._root / "objects"

    @property
    def intermediate_dir(self) -> Path:
        return self._root / "intermediate"

    @property
    def logs_dir(self) -> Path:
        return self._root / "logs"

    @property
    def outputs_dir(self) -> Path:
        return self._root / "outputs"

    @property
    def manifest(self) -> dict:
        return deepcopy(self._manifest)
