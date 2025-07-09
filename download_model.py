#!/usr/bin/env python3
"""download_huggingface_models.py
================================
Download a model repository from the Hugging Face Hub to a local directory **without** using
symbolic links.

Configure the two global variables below before running:

* ``MODEL_NAME`` – the repository ID on the Hub (e.g. ``"bert-base-uncased"``, ``"facebook/galactica-125m"``).
* ``PARENT_DIR`` – the directory **inside which** the repository files will be stored.

Example
-------
::

   MODEL_NAME = "facebook/galactica-125m"
   PARENT_DIR = "/data/hf_models"

Then execute:
::

   python download_huggingface_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import os


try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover – huggingface_hub missing
    print(
        "[ERROR] The 'huggingface_hub' package is required.\n"
        "Install it with: pip install --upgrade huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)


def download_model(model_name: str, parent_dir: str | Path) -> Path:  # noqa: D401
    """Download *model_name* into *parent_dir* (no symlinks)."""
    parent_path = Path(parent_dir).expanduser().resolve()
    parent_path.mkdir(parents=True, exist_ok=True)

    # Replace `/` to avoid creating nested directories under *parent_dir*.
    # local_dir = parent_path / model_name.replace("/", "__")
    local_dir = f"{parent_path}/{model_name}"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[INFO] Downloading '{model_name}' to '{local_dir}' …")

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # ✅ no symbolic links
        resume_download=True,          # resume if partially downloaded
        max_workers=8,                 # parallel workers for speed
        token="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"
    )

    print("[SUCCESS] Download completed!")
    return local_dir


def main(model_name) -> None:  # noqa: D401
    try:
        download_model(model_name, PARENT_DIR)
    except HfHubHTTPError as err:
        print(f"[ERROR] Failed to download '{model_name}': {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover – catch‑all
        print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # ======== USER CONFIGURATION =========
    # MODEL_NAME: str = "Qwen/Qwen2.5-Math-0.5B-Instruct"   # ← EDIT ME
    MODEL_NAME_LIST = [
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]
    FINAL_DIR: str = os.environ["MY_MODEL_DIR"]
    PARENT_DIR = "/tmp/pretrain"
    # =====================================
    for model_name in MODEL_NAME_LIST:
        main(model_name)

    # move all subdirectories under PARENT_DIR to FINAL_DIR
    os.system(f"mv {PARENT_DIR}/* {FINAL_DIR}")
    