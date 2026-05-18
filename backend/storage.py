from pathlib import Path

from .config import BASE_DIR, get_settings


settings = get_settings()


def safe_upload_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        raise ValueError("文件名不能为空")
    return name


def get_upload_path(filename: str) -> Path:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return (settings.uploads_dir / safe_upload_filename(filename)).resolve()


def resolve_document_path(file_path: str | None, filename: str | None = None) -> Path:
    raw_path = Path(file_path) if file_path else None
    raw_name = filename or (raw_path.name if raw_path else "")
    safe_name = safe_upload_filename(raw_name)

    candidates = []
    if raw_path:
        candidates.append(raw_path)

    candidates.extend(
        [
            settings.uploads_dir / safe_name,
            BASE_DIR / "uploads" / safe_name,
            BASE_DIR.parent / "uploads" / safe_name,
            BASE_DIR.parent / "data" / "uploads" / safe_name,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return get_upload_path(safe_name)
