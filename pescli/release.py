# 모델 발행 — 오너의 컴퓨터에서 ONNX 산출물을 R2 에 올리고 릴리스 메타를 등록한다 (fai publish).
#
# 서버는 모델 파일을 만지지 않는다. 브라우저는 R2(cdn.getpes.com)에서 직접 내려받고,
# 서버 API 에는 "어느 버전이 최신인가" 라는 메타데이터만 기록한다.
#
# 채널: candidate(후보) → published(발행). 평가가 이전 모델보다 나쁘면 발행하지 않는다.
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

R2_PUBLIC_BASE = "https://cdn.getpes.com"
R2_PREFIX = "fai"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest(
    version: str,
    model_path: Path,
    tokenizer_path: Path,
    *,
    docs_count: int,
    train_loss: float | None,
    val_loss: float | None,
    step: int,
) -> dict:
    """브라우저가 폴링하는 latest.json 매니페스트를 만든다.

    브라우저는 sha256 으로 다운로드 무결성을 검증한 뒤 캐시에 반영한다.
    """
    return {
        "version": version,
        "model": {
            "url": f"{R2_PUBLIC_BASE}/{R2_PREFIX}/{version}/{model_path.name}",
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
        },
        "tokenizer": {
            "url": f"{R2_PUBLIC_BASE}/{R2_PREFIX}/{version}/{tokenizer_path.name}",
            "sha256": sha256_file(tokenizer_path),
            "bytes": tokenizer_path.stat().st_size,
        },
        "docsCount": docs_count,
        "step": step,
        "trainLoss": train_loss,
        "valLoss": val_loss,
    }


def upload_to_r2(local: Path, key: str, repo_root: Path) -> None:
    """기존 scripts/r2-put.py(boto3 멀티파트)로 R2 에 업로드한다.

    R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY 환경변수가 필요하다.
    """
    script = repo_root / "scripts" / "r2-put.py"
    if not script.exists():
        raise FileNotFoundError(
            f"{script} 가 없습니다. pes 저장소 루트에서 실행하거나 --repo-root 로 지정하세요."
        )
    if not os.environ.get("R2_ACCESS_KEY_ID") or not os.environ.get("R2_SECRET_ACCESS_KEY"):
        raise RuntimeError(
            "R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY 환경변수가 필요합니다.\n"
            "  Cloudflare 대시보드 → R2 → Manage R2 API Tokens → Create(Object Read & Write)"
        )
    subprocess.run(
        ["python3", str(script), str(local), key],
        check=True,
    )


def register_release(server: str, token: str, manifest: dict, status: str = "candidate") -> dict:
    """릴리스 메타데이터를 getpes.com API 에 등록한다 (파일은 이미 R2 에 있다)."""
    import httpx

    res = httpx.post(
        f"{server}/api/fai/releases",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "version": manifest["version"],
            "modelUrl": manifest["model"]["url"],
            "tokenizerUrl": manifest["tokenizer"]["url"],
            "modelSha256": manifest["model"]["sha256"],
            "byteSize": manifest["model"]["bytes"],
            "docsCount": manifest["docsCount"],
            "trainLoss": manifest["trainLoss"],
            "valLoss": manifest["valLoss"],
            "status": status,
        },
        timeout=60,
    )
    if res.status_code not in (200, 201):
        raise RuntimeError(f"릴리스 등록 실패 (HTTP {res.status_code}): {res.text[:300]}")
    return res.json()


def write_manifest(manifest: dict, out_dir: Path) -> Path:
    path = out_dir / "latest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
