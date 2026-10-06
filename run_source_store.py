"""One run's UTF-8 source snapshots, independent of retrieval and object storage.

Each file is replaced atomically, with metadata committed last. There is no
multi-file transaction or concurrent-writer guarantee: interrupted replacement
can leave an inconsistent snapshot, which get/contains reject. Path checks do
not protect against hostile concurrent filesystem or symlink replacement.
"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from run_workspace import ResearchRunWorkspace


class RunSourceStoreError(Exception):
    """Base error, including filesystem failures."""


class RunSourceStoreValidationError(RunSourceStoreError):
    """Material or workspace paths are invalid."""


class RunSourceStoreCorruptionError(RunSourceStoreError):
    """A stored snapshot is incomplete, invalid, or fails a content hash check."""


_OPTIONAL_FIELDS = (
    "title", "source_type", "locator", "publisher", "published_date",
    "content_type", "primary_or_secondary", "independence_group",
)
_METADATA_FIELDS = {
    "source_ref", *_OPTIONAL_FIELDS, "raw_file", "normalized_file",
    "raw_sha256", "normalized_sha256", "saved_at",
}


def _source_key(source_ref: str) -> str:
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise RunSourceStoreValidationError("source_ref must be a non-blank string")
    try:
        return sha256(source_ref.encode("utf-8")).hexdigest()
    except UnicodeEncodeError as exc:
        raise RunSourceStoreValidationError("source_ref must be valid UTF-8 text") from exc


def _text_bytes(value, field: str) -> bytes:
    if not isinstance(value, str):
        raise RunSourceStoreValidationError(f"{field} must be a string")
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RunSourceStoreValidationError(f"{field} must be valid UTF-8 text") from exc


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RunSourceStore:
    def __init__(self, workspace: ResearchRunWorkspace):
        if not isinstance(workspace, ResearchRunWorkspace):
            raise RunSourceStoreValidationError("RunSourceStore requires a ResearchRunWorkspace")
        self._root = workspace.root
        self._raw_dir = workspace.sources_raw_dir
        self._normalized_dir = workspace.sources_normalized_dir
        self._check_directories()

    def _check_directories(self) -> None:
        # Keep the original canonical paths as anchors, even if links change.
        try:
            for path in (self._root, self._root / "sources", self._raw_dir, self._normalized_dir):
                if path.resolve() != path:
                    raise RunSourceStoreValidationError(f"Workspace directory redirected by symlink: {path}")
                if not path.is_dir():
                    raise RunSourceStoreValidationError(f"Required workspace directory is missing: {path}")
        except (RuntimeError, ValueError) as exc:
            raise RunSourceStoreValidationError(f"Invalid workspace directory: {exc}") from exc
        except OSError as exc:
            raise RunSourceStoreError(f"Cannot inspect workspace directories: {exc}") from exc

    def _check_path(self, path: Path) -> Path:
        self._check_directories()
        boundary = path.parent
        if boundary not in (self._raw_dir, self._normalized_dir):
            raise RunSourceStoreValidationError(f"Path is outside source storage directories: {path}")
        try:
            resolved = path.resolve()
        except (RuntimeError, ValueError) as exc:
            raise RunSourceStoreValidationError(f"Cannot resolve source path {path}: {exc}") from exc
        except OSError as exc:
            raise RunSourceStoreError(f"Cannot inspect source path {path}: {exc}") from exc
        if resolved == boundary or not resolved.is_relative_to(boundary):
            raise RunSourceStoreValidationError(f"Source path escapes its workspace boundary: {path}")
        return path

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        return (
            self._check_path(self._raw_dir / f"{key}.raw"),
            self._check_path(self._normalized_dir / f"{key}.txt"),
            self._check_path(self._normalized_dir / f"{key}.json"),
        )

    def _exists(self, path: Path) -> bool:
        try:
            self._check_path(path).lstat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise RunSourceStoreError(f"Cannot inspect source file {path}: {exc}") from exc
        return True

    def _read(self, path: Path) -> bytes:
        try:
            return self._check_path(path).read_bytes()
        except FileNotFoundError as exc:
            raise RunSourceStoreCorruptionError(f"Snapshot file is missing: {path.name}") from exc
        except OSError as exc:
            raise RunSourceStoreError(f"Cannot read source file {path}: {exc}") from exc

    def _validate_metadata(self, metadata, source_ref: str, raw: Path, normalized: Path) -> None:
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
            raise RunSourceStoreCorruptionError("Snapshot metadata has invalid fields")
        if metadata["source_ref"] != source_ref:
            raise RunSourceStoreCorruptionError("Snapshot metadata source_ref does not match")
        if any(metadata[field] is not None and not isinstance(metadata[field], str) for field in _OPTIONAL_FIELDS):
            raise RunSourceStoreCorruptionError("Snapshot metadata values must be strings or null")
        if metadata["normalized_file"] != str(normalized.relative_to(self._root)):
            raise RunSourceStoreCorruptionError("Snapshot normalized_file does not match the expected path")
        hashes = [metadata["normalized_sha256"]]
        if metadata["raw_file"] is None:
            if metadata["raw_sha256"] is not None:
                raise RunSourceStoreCorruptionError("Absent raw_file must have a null raw_sha256")
        else:
            if metadata["raw_file"] != str(raw.relative_to(self._root)):
                raise RunSourceStoreCorruptionError("Snapshot raw_file does not match the expected path")
            hashes.append(metadata["raw_sha256"])
        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise RunSourceStoreCorruptionError("Snapshot metadata has an invalid SHA-256 digest")
        try:
            saved_at = datetime.fromisoformat(metadata["saved_at"])
        except (TypeError, ValueError) as exc:
            raise RunSourceStoreCorruptionError("Snapshot saved_at must be an ISO timestamp") from exc
        if saved_at.utcoffset() != timedelta(0):
            raise RunSourceStoreCorruptionError("Snapshot saved_at must have a UTC timezone")

    def _content(self, path: Path, expected_hash: str) -> str:
        data = self._read(path)
        if sha256(data).hexdigest() != expected_hash:
            raise RunSourceStoreCorruptionError(f"SHA-256 mismatch for {path.name}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RunSourceStoreCorruptionError(f"Invalid UTF-8 in {path.name}") from exc

    def get(self, source_ref: str) -> dict | None:
        """Return validated metadata and exact text; absent raw_content is omitted."""
        raw, normalized, metadata_path = self._paths(_source_key(source_ref))
        if not self._exists(metadata_path):
            if self._exists(raw) or self._exists(normalized):
                raise RunSourceStoreCorruptionError("Snapshot metadata is missing for existing content files")
            return None
        try:
            metadata = json.loads(self._read(metadata_path).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunSourceStoreCorruptionError("Snapshot metadata is not valid UTF-8 JSON") from exc
        self._validate_metadata(metadata, source_ref, raw, normalized)
        material = {**metadata, "content": self._content(normalized, metadata["normalized_sha256"])}
        if metadata["raw_file"] is not None:
            material["raw_content"] = self._content(raw, metadata["raw_sha256"])
        elif self._exists(raw):
            raise RunSourceStoreCorruptionError("Snapshot has an unreferenced raw content file")
        return material

    def contains(self, source_ref: str) -> bool:
        """Corrupt snapshots raise; they are never reported as present or absent."""
        return self.get(source_ref) is not None

    def _discard_temporary(self, path: Path) -> None:
        try:
            self._check_path(path).unlink(missing_ok=True)
        except (OSError, RunSourceStoreError):
            # Keep cleanup within the boundary and preserve the original error.
            pass

    def _stage(self, path: Path, data: bytes) -> Path:
        self._check_path(path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                self._check_path(temporary_path)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            return temporary_path
        except Exception:
            if temporary_path is not None:
                self._discard_temporary(temporary_path)
            raise

    def put(self, material: dict) -> str:
        """Save a whitelist snapshot, preserving identical snapshots and saved_at.

        content/raw_content are strings, including empty strings. If raw_content
        is omitted, no raw file is retained. Existing snapshots must pass get's
        integrity checks before they can be replaced.
        """
        if not isinstance(material, dict):
            raise RunSourceStoreValidationError("material must be a dict")
        source_ref = material.get("source_ref")
        key = _source_key(source_ref)
        normalized_bytes = _text_bytes(material.get("content"), "content")
        raw_bytes = _text_bytes(material["raw_content"], "raw_content") if "raw_content" in material else None
        optional = {field: material.get(field) for field in _OPTIONAL_FIELDS}
        if any(value is not None and not isinstance(value, str) for value in optional.values()):
            raise RunSourceStoreValidationError("Metadata fields must be strings or None")
        raw, normalized, metadata_path = self._paths(key)
        metadata = {
            "source_ref": source_ref, **optional,
            "raw_file": str(raw.relative_to(self._root)) if raw_bytes is not None else None,
            "normalized_file": str(normalized.relative_to(self._root)),
            "raw_sha256": sha256(raw_bytes).hexdigest() if raw_bytes is not None else None,
            "normalized_sha256": sha256(normalized_bytes).hexdigest(),
        }
        existing = self.get(source_ref)
        if existing is not None and all(existing[field] == value for field, value in metadata.items()):
            return key
        metadata["saved_at"] = _now_utc().isoformat()
        try:
            metadata_bytes = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        except (ValueError, TypeError) as exc:
            raise RunSourceStoreValidationError(f"Cannot encode snapshot metadata: {exc}") from exc

        staged = []
        try:
            contents = ([(raw, raw_bytes)] if raw_bytes is not None else []) + [(normalized, normalized_bytes), (metadata_path, metadata_bytes)]
            for path, data in contents:
                staged.append((self._stage(path, data), path))
            for temporary_path, path in staged[:-1]:
                os.replace(self._check_path(temporary_path), self._check_path(path))
            if raw_bytes is None:
                self._check_path(raw).unlink(missing_ok=True)
            temporary_path, path = staged[-1]
            os.replace(self._check_path(temporary_path), self._check_path(path))
        except OSError as exc:
            raise RunSourceStoreError(f"Cannot save source snapshot {key}: {exc}") from exc
        finally:
            for temporary_path, _ in staged:
                self._discard_temporary(temporary_path)
        return key
