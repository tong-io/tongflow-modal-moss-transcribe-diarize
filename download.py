"""Modal download entry for MOSS-Transcribe-Diarize 0.9B.

Run:
  modal run download.py::download

Fetches the whole repo (weights + the remote code the model loads with
``trust_remote_code=True``) onto the shared ``models`` volume, so the container
never reaches Hugging Face at request time. ~2 GB.

OpenMOSS-Team/MOSS-Transcribe-Diarize is public and Apache-2.0; HF_TOKEN is
only needed if it becomes gated.
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}

REPO_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MODEL_DIR = f"/models/{REPO_ID}"

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")

_download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.34.0,<1.0"
)


@model_downloader.function(
    image=_download_image,
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download(repo_id: str, dest: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    if os.path.isdir(dest) and os.path.isfile(os.path.join(dest, "config.json")):
        print(f"skip (exists): {dest}")
        return
    print(f"Downloading {repo_id} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except GatedRepoError as e:
        raise RuntimeError(
            f"{repo_id} is gated and this HF_TOKEN is not authorized. Open "
            f"https://huggingface.co/{repo_id}, accept the license, then re-run."
        ) from e
    volume.commit()
    print("Done.")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote(REPO_ID, MODEL_DIR)
