"""Shared environment setup for local, VS Code Colab and browser Colab runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


DEFAULT_DRIVE_PROJECT = "MyDrive/PhysioLoad_AI_Project"
DEFAULT_VERSION = "v1"
KNOWN_USERS = {
    "berk-can": {"berk can", "berk can sefer", "berkcan", "brcnsfr6104@gmail.com"},
}


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def in_vscode() -> bool:
    return bool(os.environ.get("VSCODE_PID") or os.environ.get("TERM_PROGRAM") == "vscode")


def default_num_workers() -> int:
    return 0 if platform.system() in {"Windows", "Darwin"} else min(4, os.cpu_count() or 1)


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def find_local_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    for path in (candidate, *candidate.parents):
        if (path / "physioload_env.py").exists() and (path / "physioload").is_dir():
            return path
    raise FileNotFoundError("physioload_env.py was not found above the current directory")


def _normalize_user(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("_", "-")
    for folder, aliases in KNOWN_USERS.items():
        accepted = {folder, *(alias.lower() for alias in aliases)}
        if normalized in accepted or value.strip().lower() in accepted:
            return folder
    return None


def _git_identity(root: Path) -> str | None:
    for key in ("user.email", "user.name"):
        try:
            value = subprocess.check_output(
                ["git", "-C", str(root), "config", key], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        resolved = _normalize_user(value)
        if resolved:
            return resolved
    return None


def resolve_user(user: str | None, root: Path, *, allow_unknown: bool = False) -> str:
    explicit = user or os.environ.get("PHYSIOLOAD_USER")
    if explicit:
        resolved = _normalize_user(explicit)
        if resolved:
            return resolved
        if allow_unknown:
            return explicit.strip().lower().replace(" ", "-")
        raise ValueError(f"unknown user {explicit!r}; known users: {', '.join(KNOWN_USERS)}")
    identity_file = root / ".physioload_user"
    if not in_colab() and identity_file.exists():
        resolved = _normalize_user(identity_file.read_text(encoding="utf-8"))
        if resolved:
            return resolved
    return _git_identity(root) or "berk-can"


def fingerprint(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def check_manifest(directory: Path, params: dict[str, Any]) -> tuple[str, str]:
    path = directory / "manifest.json"
    if not path.exists():
        if directory.exists() and any(directory.iterdir()):
            return "unmanaged", "data exists but manifest.json is missing"
        return "missing", "no manifest yet"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = fingerprint(params)
    if manifest.get("fingerprint") == expected:
        return "match", f"manifest matches {expected}"
    return "mismatch", "parameters differ; create a new processed-data version"


def write_manifest(directory: Path, params: dict[str, Any], *, version: str, counts: dict | None = None) -> dict:
    status, message = check_manifest(directory, params)
    if status in {"mismatch", "unmanaged"}:
        raise ValueError(message)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "fingerprint": fingerprint(params),
        "params": params,
        "counts": counts or {},
    }
    path = directory / "manifest.json"
    if not path.exists():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


@dataclass(frozen=True)
class Environment:
    root: Path
    artifacts_root: Path
    data_root: Path
    user: str
    processed_version: str
    device: str
    context: str

    @property
    def user_root(self) -> Path:
        return self.artifacts_root / "users" / self.user if self.artifacts_root != self.root else self.root

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "data" / "raw"

    @property
    def metadata_dir(self) -> Path:
        return self.data_root / "data" / "metadata"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "data" / "processed" / self.processed_version

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "data" / "splits" / self.processed_version

    @property
    def checkpoints_dir(self) -> Path:
        return self.user_root / "models" / "checkpoints"

    @property
    def calibrators_dir(self) -> Path:
        return self.user_root / "models" / "calibrators"

    @property
    def outputs_dir(self) -> Path:
        return self.user_root / "outputs"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    def with_user(self, user: str) -> "Environment":
        updated = replace(self, user=resolve_user(user, self.root))
        updated.ensure_directories()
        return updated

    def with_version(self, version: str) -> "Environment":
        if not version or any(char in version for char in "/\\"):
            raise ValueError("processed version must be a simple directory name")
        updated = replace(self, processed_version=version)
        updated.ensure_directories()
        return updated

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_dir / "fit3d",
            self.raw_dir / "human36m",
            self.raw_dir / "other",
            self.metadata_dir,
            self.interim_dir / "static_frames",
            self.processed_dir,
            self.splits_dir,
            self.checkpoints_dir,
            self.calibrators_dir,
            self.predictions_dir,
            self.metrics_dir,
            self.figures_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        return "\n".join(
            [
                f"context       : {self.context}",
                f"user          : {self.user}",
                f"processed ver : {self.processed_version}",
                f"device        : {self.device}",
                f"code root     : {self.root}",
                f"artifacts root: {self.artifacts_root}",
                f"data root     : {self.data_root}",
            ]
        )


def setup_environment(
    *,
    root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    data_root: str | Path | None = None,
    user: str | None = None,
    processed_version: str | None = None,
) -> Environment:
    project_root = Path(root).resolve() if root else find_local_root()
    artifact_path = Path(artifacts_root).resolve() if artifacts_root else project_root
    data_path = Path(data_root).resolve() if data_root else artifact_path
    version = processed_version or os.environ.get("PHYSIOLOAD_PROCESSED_VERSION", DEFAULT_VERSION)
    context = "colab (vscode)" if in_colab() and in_vscode() else "colab" if in_colab() else "vscode local" if in_vscode() else "local"
    env = Environment(
        root=project_root,
        artifacts_root=artifact_path,
        data_root=data_path,
        user=resolve_user(user, project_root),
        processed_version=version,
        device=get_device(),
        context=context,
    )
    env.ensure_directories()
    return env


def user_selector(env: Environment):
    try:
        import ipywidgets as widgets
    except ImportError:
        return env.user
    options = [(name, name) for name in KNOWN_USERS]
    return widgets.Dropdown(options=options, value=env.user, description="user:")


def selected_user(selector, env: Environment) -> str:
    value = getattr(selector, "value", selector) or env.user
    return resolve_user(str(value), env.root)


def version_selector(env: Environment):
    versions_root = env.data_root / "data" / "processed"
    versions = sorted(path.name for path in versions_root.iterdir() if path.is_dir())
    if env.processed_version not in versions:
        versions.append(env.processed_version)
    try:
        import ipywidgets as widgets
    except ImportError:
        return env.processed_version
    return widgets.Dropdown(options=versions, value=env.processed_version, description="version:")


def selected_version(selector, env: Environment) -> str:
    return str(getattr(selector, "value", selector) or env.processed_version)


def flush_drive() -> None:
    if in_colab():
        try:
            from google.colab import drive

            drive.flush_and_unmount()
        except Exception as error:  # pragma: no cover - Colab only
            print("Drive flush failed:", error)

