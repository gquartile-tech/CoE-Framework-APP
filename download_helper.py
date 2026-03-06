from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict


OUT_DIR = Path("/mnt/data")
MIN_BYTES = 20000


def safe_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "UNKNOWN_ACCOUNT"

    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name or "UNKNOWN_ACCOUNT"


def assert_file_ok(path: str | Path, min_bytes: int = MIN_BYTES) -> None:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    size = p.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"File too small ({size} bytes): {p}")


def create_download_artifact(
    canonical_path: str | Path,
    hash_name: str,
    out_dir: str | Path = OUT_DIR,
    min_bytes: int = MIN_BYTES,
) -> Dict[str, str]:
    """
    Create a fresh timestamped .xlsm artifact in /mnt/data and return
    the exact sandbox link that must be used in the final response.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    canonical = Path(canonical_path)
    safe_hash = safe_filename(hash_name)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    download_filename = f"{safe_hash} - Framework Analysis - {ts}.xlsm"
    download_path = out_path / download_filename

    # Validate canonical file before copying
    assert_file_ok(canonical, min_bytes=min_bytes)

    # Create timestamped artifact
    shutil.copyfile(canonical, download_path)

    # Validate copied artifact
    assert_file_ok(download_path, min_bytes=min_bytes)

    # Extra safeguard: ensure the file is really visible in the filesystem
    if not download_path.exists():
        raise RuntimeError(f"Download artifact was not created: {download_path}")

    sandbox_link = f"sandbox:{download_path.as_posix()}"

    return {
        "canonical_path": str(canonical),
        "download_path": str(download_path),
        "download_filename": download_filename,
        "sandbox_link": sandbox_link,
    }


def print_sandbox_link(info: Dict[str, str]) -> None:
    print(f"SANDBOX_LINK: {info['sandbox_link']}")
    print(f"DOWNLOAD_MARKDOWN: [Download the Framework Analysis file]({info['sandbox_link']})")
    print(f"DOWNLOAD_FILENAME: {info['download_filename']}")
