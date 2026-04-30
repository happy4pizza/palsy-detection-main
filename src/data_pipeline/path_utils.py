from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_manifest_filepath(
    filepath: str | Path,
    *,
    project_root: Path | None = None,
) -> Path:
    base_dir = (project_root or PROJECT_ROOT).resolve()
    path = Path(filepath)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def relativize_to_project_root(
    path: str | Path,
    *,
    project_root: Path | None = None,
) -> str:
    base_dir = (project_root or PROJECT_ROOT).resolve()
    resolved_path = Path(path).resolve()
    try:
        return str(resolved_path.relative_to(base_dir))
    except ValueError:
        return str(resolved_path)
