#!/usr/bin/env python3
"""FLORA download: small files via hub, large shards via curl -C - (official HF).

CLI entry lives under ``eai/scripts/`` (not ``eai/log/``). Logs go to ``eai/log/``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

MODELS = Path("/share_data/projects/mahjong/share/personal/liyichao/models")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
os.environ.setdefault("HF_HOME", str(MODELS / "Flora" / "hf_home"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

JOBS = [
    ("nvidia/Flora-TempSeg", MODELS / "Flora-TempSeg"),
    ("nvidia/Flora-Captioner", MODELS / "Flora-Captioner"),
    ("nvidia/Flora-Rewarder", MODELS / "Flora-Rewarder"),
    ("Qwen/Qwen3.5-9B", MODELS / "Qwen" / "Qwen3.5-9B"),
]

BIG_SUFFIX = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


def log(msg: str) -> None:
    print(msg, flush=True)


def local_bytes(dest: Path) -> int:
    if not dest.exists():
        return 0
    total = 0
    for p in dest.rglob("*"):
        if p.is_file() and ".cache" not in p.parts:
            total += p.stat().st_size
    return total


def repo_complete(repo: str, dest: Path, api: HfApi) -> bool:
    info = api.repo_info(repo, files_metadata=True, token=TOKEN)
    expected = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    if not expected:
        return False
    for name, size in expected.items():
        path = dest / name
        if not path.is_file():
            return False
        if size and path.stat().st_size != size:
            return False
    return True


def curl_download(repo: str, rel: str, dest: Path, size: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    # encode path segments but keep /
    encoded = "/".join(quote(part, safe="") for part in rel.split("/"))
    url = f"{ENDPOINT}/{repo}/resolve/main/{encoded}"
    headers = []
    if TOKEN:
        headers.extend(["-H", f"Authorization: Bearer {TOKEN}"])
    # Resume with -C -; follow redirects; long timeouts for huge shards
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "60",
        "--speed-limit",
        "10000",  # abort if <10KB/s for 60s
        "-C",
        "-",
        "-o",
        str(out),
        *headers,
        url,
    ]
    log(f"  curl → {rel} (expect {size/1e9:.2f} GB)")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(err[-1] if err else f"curl exit={proc.returncode}")
    got = out.stat().st_size if out.is_file() else -1
    if size and got != size:
        raise RuntimeError(f"size mismatch {got} != {size}")
    dt = max(time.time() - t0, 1e-3)
    log(f"  curl OK {got} bytes @ {got/dt/1e6:.1f} MB/s")


def download_repo(repo: str, dest: Path, api: HfApi) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log(f"==== {time.strftime('%F %T')} {repo} -> {dest} ====")
    if repo_complete(repo, dest, api):
        log(f"SKIP complete {repo} ({local_bytes(dest)/1e9:.2f} GB)")
        return
    info = api.repo_info(repo, files_metadata=True, token=TOKEN)
    files = sorted(
        [(s.rfilename, s.size or 0) for s in (info.siblings or [])],
        key=lambda x: (1 if x[0].endswith(BIG_SUFFIX) else 0, x[0]),
    )
    for i, (rel, size) in enumerate(files, 1):
        path = dest / rel
        if path.is_file() and size and path.stat().st_size == size:
            log(f"[{i}/{len(files)}] SKIP {rel}")
            continue
        attempt = 0
        while True:
            attempt += 1
            try:
                log(f"[{i}/{len(files)}] GET {rel} ({size/1e9:.2f} GB) attempt={attempt}")
                if rel.endswith(BIG_SUFFIX) and size >= 50_000_000:
                    curl_download(repo, rel, dest, size)
                else:
                    t0 = time.time()
                    out = hf_hub_download(
                        repo_id=repo,
                        filename=rel,
                        local_dir=str(dest),
                        token=TOKEN,
                    )
                    got = Path(out).stat().st_size if Path(out).is_file() else -1
                    dt = max(time.time() - t0, 1e-3)
                    log(f"[{i}/{len(files)}] OK {rel} ({got} bytes, {got/dt/1e6:.1f} MB/s)")
                break
            except Exception as exc:  # noqa: BLE001
                wait = min(180, 15 * attempt)
                log(f"[{i}/{len(files)}] FAIL {rel}: {type(exc).__name__}: {exc}")
                log(f"  sleep {wait}s then retry (attempt {attempt})")
                # remove truncated target so curl can restart cleanly if needed
                if path.is_file() and size and path.stat().st_size != size and attempt % 3 == 0:
                    log(f"  remove partial {path}")
                    path.unlink(missing_ok=True)
                time.sleep(wait)


def main() -> int:
    api = HfApi()
    log(f"endpoint={ENDPOINT} token={'yes' if TOKEN else 'no'}")
    for repo, dest in JOBS:
        download_repo(repo, dest, api)
    log(f"==== ALL DONE {time.strftime('%F %T')} ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
