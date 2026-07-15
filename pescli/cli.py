# fai — 파이 인공지능(FAI) 로컬 학습 CLI.
#
#   fai login  : getpes.com 개발자 토큰 등록
#   fai pull   : 학습 데이터 스냅샷 다운로드 → 토크나이저 학습 → 바이너리 데이터셋 생성
#   fai train  : 이 컴퓨터에서 트레이닝 (GPU/CPU 자동 감지)
#   fai push   : 훈련 결과 체크포인트만 서버에 업로드
#   fai build  : (오너용) ONNX 변환 + int8 양자화 — R2 업로드 준비
#   fai merge  : (오너용) 수집된 체크포인트들을 로컬 FedAvg 병합
#
# 서버는 인증·DB 입출력만 한다. 무거운 연산은 전부 이 명령들이 실행되는 컴퓨터에서 일어난다.
from __future__ import annotations

from pathlib import Path

import click
import httpx

from .config import CKPT_DIR, DATA_DIR, BUILD_DIR, CliConfig, ensure_dirs

SNAPSHOT_PATH = DATA_DIR / "snapshot.jsonl"
SAMPLES_PATH = DATA_DIR / "samples.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CKPT_PATH = CKPT_DIR / "ckpt.pt"


@click.group()
def main() -> None:
    """파이 인공지능(FAI) — 내 컴퓨터로 함께 키우는 인공지능."""
    ensure_dirs()


@main.command()
@click.option("--token", prompt="getpes.com 개발자 토큰", hide_input=True, help="https://getpes.com/ai/fai 에서 발급")
@click.option("--server", default=None, help="서버 주소 (기본: https://getpes.com)")
def login(token: str, server: str | None) -> None:
    """개발자 토큰을 ~/.fai/config.json 에 저장한다."""
    cfg = CliConfig.load()
    cfg.token = token.strip()
    if server:
        cfg.server = server.rstrip("/")
    cfg.save()
    click.echo("[FAI] 토큰 저장 완료.")


def _require_token() -> CliConfig:
    cfg = CliConfig.load()
    if not cfg.token:
        raise click.ClickException("토큰이 없습니다. 먼저 `fai login` 을 실행하세요.")
    return cfg


@main.command()
@click.option("--token", default=None, help="토큰을 직접 지정 (login 생략 가능)")
def pull(token: str | None) -> None:
    """학습 데이터 스냅샷을 내려받고, 토크나이저·바이너리 데이터셋을 이 컴퓨터에서 생성한다."""
    cfg = CliConfig.load()
    if token:
        cfg.token = token.strip()
        cfg.save()
    cfg = _require_token()

    click.echo(f"[FAI] 스냅샷 다운로드: {cfg.server}/api/fai/snapshot")
    with httpx.stream(
        "GET",
        f"{cfg.server}/api/fai/snapshot",
        headers={"Authorization": f"Bearer {cfg.token}"},
        timeout=300,
    ) as res:
        if res.status_code != 200:
            raise click.ClickException(f"다운로드 실패 (HTTP {res.status_code}) — 토큰을 확인하세요.")
        with SNAPSHOT_PATH.open("wb") as f:
            for chunk in res.iter_bytes():
                f.write(chunk)

    # 전처리는 전부 이 컴퓨터에서 (서버 금지 원칙)
    from .dataset import build_bin_dataset, snapshot_to_samples
    from .korean_tokenizer import load_tokenizer, train_korean_tokenizer

    docs = snapshot_to_samples(SNAPSHOT_PATH, SAMPLES_PATH)
    if docs == 0:
        raise click.ClickException("스냅샷에 문서가 없습니다. 데이터 기여가 더 필요합니다.")
    click.echo(f"[FAI] 문서 {docs}건 → {SAMPLES_PATH}")

    if TOKENIZER_PATH.exists():
        click.echo("[FAI] 기존 토크나이저 재사용 (새로 학습하려면 data/tokenizer.json 삭제)")
        tokenizer = load_tokenizer(TOKENIZER_PATH)
    else:
        click.echo("[FAI] 한국어 ByteLevel BPE 토크나이저 학습 중…")
        tokenizer = train_korean_tokenizer(SAMPLES_PATH, TOKENIZER_PATH)

    n_train, n_val = build_bin_dataset(SAMPLES_PATH, tokenizer, DATA_DIR)
    click.echo(f"[FAI] 데이터셋 완료: train {n_train:,} / val {n_val:,} 토큰")
    click.echo("[FAI] 다음 단계: fai train")


@main.command()
@click.option("--steps", default=5000, show_default=True, help="이번 세션의 학습 스텝 수")
@click.option("--batch-size", default=16, show_default=True)
@click.option("--fresh", is_flag=True, help="기존 체크포인트를 무시하고 처음부터 학습")
def train(steps: int, batch_size: int, fresh: bool) -> None:
    """이 컴퓨터에서 파이를 트레이닝한다 (MPS/CUDA/CPU 자동 감지, Ctrl+C 안전 중단)."""
    if not (DATA_DIR / "train.bin").exists():
        raise click.ClickException("데이터셋이 없습니다. 먼저 `fai pull` 을 실행하세요.")

    from .train import TrainConfig, train as run_train

    result = run_train(
        data_dir=DATA_DIR,
        ckpt_path=CKPT_PATH,
        resume_from=None if fresh else CKPT_PATH,
        tc=TrainConfig(max_steps=steps, batch_size=batch_size),
    )
    click.echo(
        f"[FAI] 완료: {result['steps']} steps | train {result['train_loss']:.4f} | "
        f"val {result['val_loss']:.4f} | {result['device']}"
    )
    click.echo("[FAI] 다음 단계: fai push (결과만 서버로 업로드)")


@main.command()
def push() -> None:
    """훈련 결과 체크포인트만 getpes.com 에 업로드한다 (병합·발행은 오너가 로컬에서)."""
    cfg = _require_token()
    if not CKPT_PATH.exists():
        raise click.ClickException("체크포인트가 없습니다. 먼저 `fai train` 을 실행하세요.")

    import torch

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    from .train import pick_device

    fields = {
        "steps_trained": str(int(ckpt.get("step", 0))),
        "device_type": pick_device(),
    }
    if ckpt.get("train_loss") is not None:
        fields["train_loss"] = f"{float(ckpt['train_loss']):.4f}"
    if ckpt.get("val_loss") is not None:
        fields["val_loss"] = f"{float(ckpt['val_loss']):.4f}"

    size_mb = CKPT_PATH.stat().st_size / 1024 / 1024
    click.echo(f"[FAI] 업로드 중: {CKPT_PATH.name} ({size_mb:.1f}MB)")
    with CKPT_PATH.open("rb") as f:
        res = httpx.post(
            f"{cfg.server}/api/fai/checkpoints",
            headers={"Authorization": f"Bearer {cfg.token}"},
            data=fields,
            files={"file": (CKPT_PATH.name, f, "application/octet-stream")},
            timeout=600,
        )
    if res.status_code != 201:
        raise click.ClickException(f"업로드 실패 (HTTP {res.status_code}): {res.text[:200]}")
    body = res.json()
    click.echo(f"[FAI] 업로드 완료: id={body['id']} sha256={body['sha256'][:12]}…")
    click.echo("[FAI] 고맙습니다! 오너가 검증·병합 후 다음 모델 버전에 반영합니다.")


@main.command()
@click.argument("ckpt_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=False)
@click.option("--out", type=click.Path(path_type=Path), default=None, help="병합 결과 경로")
def merge(ckpt_dir: Path | None, out: Path | None) -> None:
    """(오너용) 수집한 체크포인트(*.pt)들을 steps 가중 FedAvg 로 병합한다."""
    src = ckpt_dir or CKPT_DIR
    paths = sorted(src.glob("*.pt"))
    if not paths:
        raise click.ClickException(f"{src} 에 .pt 파일이 없습니다.")

    from .merge import merge_checkpoints

    result = merge_checkpoints(paths, out or (CKPT_DIR / "merged.pt"))
    click.echo(f"[FAI] 병합 {result['merged']}건 (총 {result['total_steps']} steps) → {result['out']}")
    for name, reason in result["rejected"]:
        click.echo(f"[FAI] 제외: {name} — {reason}")


@main.command()
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), default=None, help="변환할 체크포인트")
def build(ckpt: Path | None) -> None:
    """(오너용) ONNX 변환 + int8 양자화. 산출물을 R2(cdn.getpes.com/fai/)에 업로드하면 발행 완료."""
    src = ckpt or CKPT_PATH
    if not src.exists():
        raise click.ClickException("체크포인트가 없습니다. 먼저 `fai train` 또는 `fai merge` 를 실행하세요.")

    try:
        from .export_onnx import export_onnx
    except ImportError as e:
        raise click.ClickException(f"빌드 의존성이 없습니다. `uv sync --extra build` 를 실행하세요. ({e})")

    result = export_onnx(src, BUILD_DIR)
    click.echo(f"[FAI] fp32: {result['fp32_path']} ({result['fp32_bytes'] / 1e6:.1f}MB)")
    click.echo(f"[FAI] int8: {result['int8_path']} ({result['int8_bytes'] / 1e6:.1f}MB)")
    click.echo(f"[FAI] logits 최대 오차: {result['max_diff']:.2e} (검증 통과)")
    click.echo("[FAI] 발행: int8 모델 + data/tokenizer.json + version.json 을 R2 에 업로드하세요.")


if __name__ == "__main__":
    main()
