# fai — 파이 인공지능(FAI) 로컬 학습 CLI.
#
#   fai doctor : 이 컴퓨터가 학습 가능한지 점검 (가장 먼저 실행)
#   fai login  : getpes.com 개발자 토큰 등록
#   fai pull   : 학습 데이터 스냅샷 다운로드 → 토크나이저 → 바이너리 데이터셋 생성
#   fai train  : 이 컴퓨터에서 트레이닝 (GPU/CPU 자동 감지)
#   fai push   : 훈련 결과 체크포인트만 서버에 업로드
#   fai merge  : (오너용) 수집된 체크포인트를 로컬 FedAvg 병합
#   fai build  : (오너용) ONNX 변환 + int8 양자화
#   fai publish: (오너용) R2 업로드 + 릴리스 등록 (후보 → 발행)
#
# 서버는 인증·DB 입출력만 한다. 무거운 연산은 전부 이 명령들이 실행되는 컴퓨터에서 일어난다.
# 체크포인트 포맷은 SafeTensors — 신뢰할 수 없는 파일을 열어도 코드가 실행되지 않는다.
from __future__ import annotations

from pathlib import Path

import click
import httpx

from .checkpoint_io import CKPT_SUFFIX
from .config import BUILD_DIR, CKPT_DIR, DATA_DIR, CliConfig, ensure_dirs

SNAPSHOT_PATH = DATA_DIR / "snapshot.jsonl"
SAMPLES_PATH = DATA_DIR / "samples.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CKPT_PATH = CKPT_DIR / f"ckpt{CKPT_SUFFIX}"


@click.group()
def main() -> None:
    """파이 인공지능(FAI) — 내 컴퓨터로 함께 키우는 인공지능."""
    ensure_dirs()


@main.command()
@click.option("--speed", is_flag=True, help="실제 학습 속도를 측정해 예상 시간을 안내")
def doctor(speed: bool) -> None:
    """이 컴퓨터가 파이 트레이닝을 할 수 있는지 점검한다."""
    from .doctor import estimate_speed, run_checks

    click.echo("[FAI] 환경 점검\n")
    fatal = False
    for c in run_checks():
        mark = click.style("✓", fg="green") if c.ok else click.style("✗", fg="red" if c.fatal else "yellow")
        click.echo(f"  {mark} {c.name:12s} {c.detail}")
        fatal = fatal or (c.fatal and not c.ok)

    if speed:
        click.echo("\n[FAI] 학습 속도 측정 중…")
        result = estimate_speed()
        click.echo(f"  {result}" if result else "  측정 실패 (PyTorch 확인 필요)")

    if fatal:
        raise click.ClickException("학습을 시작할 수 없습니다. 위의 ✗ 항목을 먼저 해결하세요.")
    click.echo(click.style("\n[FAI] 학습 준비 완료!", fg="green"))


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
    click.echo("[FAI] 토큰 저장 완료. 다음: fai doctor")


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

    # 전처리는 전부 이 컴퓨터에서 (서버 무연산 원칙)
    from .dataset import build_bin_dataset, snapshot_to_samples
    from .korean_tokenizer import load_tokenizer, train_korean_tokenizer

    docs = snapshot_to_samples(SNAPSHOT_PATH, SAMPLES_PATH)
    if docs == 0:
        raise click.ClickException("스냅샷에 문서가 없습니다. 데이터 기여가 더 필요합니다.")
    click.echo(f"[FAI] 문서 {docs}건 → {SAMPLES_PATH}")

    # ⚠️ 토크나이저는 모든 참여자가 동일해야 가중치를 합칠 수 있다.
    #    서버가 발행한 토크나이저가 있으면 그것을 쓰고, 없을 때만(=최초 모델 제작) 학습한다.
    if TOKENIZER_PATH.exists():
        click.echo("[FAI] 기존 토크나이저 재사용")
        tokenizer = load_tokenizer(TOKENIZER_PATH)
    else:
        released = _fetch_released_tokenizer(cfg.server)
        if released:
            TOKENIZER_PATH.write_bytes(released)
            click.echo("[FAI] 발행된 토크나이저 다운로드 완료 (fai-ko-v1)")
            tokenizer = load_tokenizer(TOKENIZER_PATH)
        else:
            click.echo("[FAI] 발행된 토크나이저 없음 → 한국어 ByteLevel BPE 학습 (최초 모델 제작)")
            tokenizer = train_korean_tokenizer(SAMPLES_PATH, TOKENIZER_PATH)

    n_train, n_val = build_bin_dataset(SAMPLES_PATH, tokenizer, DATA_DIR)
    click.echo(f"[FAI] 데이터셋 완료: train {n_train:,} / val {n_val:,} 토큰")
    click.echo("[FAI] 다음 단계: fai train")


def _fetch_released_tokenizer(server: str) -> bytes | None:
    """발행된 최신 릴리스의 토크나이저를 받아온다 (없으면 None)."""
    try:
        res = httpx.get(f"{server}/api/fai/releases/latest", timeout=30)
        if res.status_code != 200:
            return None
        url = res.json().get("tokenizerUrl")
        if not url:
            return None
        tok = httpx.get(url, timeout=120, follow_redirects=True)
        return tok.content if tok.status_code == 200 else None
    except Exception:
        return None


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

    from .checkpoint_io import load_checkpoint
    from .train import pick_device

    _, meta = load_checkpoint(CKPT_PATH)
    fields = {
        "steps_trained": str(int(meta.get("step", 0))),
        "device_type": pick_device(),
        "format": "safetensors",
    }
    if meta.get("train_loss") is not None:
        fields["train_loss"] = f"{meta['train_loss']:.4f}"
    if meta.get("val_loss") is not None:
        fields["val_loss"] = f"{meta['val_loss']:.4f}"

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
    """(오너용) 수집한 체크포인트(*.safetensors)들을 steps 가중 FedAvg 로 병합한다."""
    src = ckpt_dir or CKPT_DIR
    paths = sorted(src.glob(f"*{CKPT_SUFFIX}"))
    if not paths:
        raise click.ClickException(f"{src} 에 {CKPT_SUFFIX} 파일이 없습니다.")

    from .merge import merge_checkpoints

    result = merge_checkpoints(paths, out or (CKPT_DIR / f"merged{CKPT_SUFFIX}"))
    click.echo(f"[FAI] 병합 {result['merged']}건 (총 {result['total_steps']} steps) → {result['out']}")
    for name, reason in result["rejected"]:
        click.echo(f"[FAI] 제외: {name} — {reason}")


@main.command()
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), default=None, help="변환할 체크포인트")
def build(ckpt: Path | None) -> None:
    """(오너용) ONNX 변환 + int8 양자화. 다음: fai publish 로 R2 발행."""
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
    click.echo("[FAI] 다음 단계: fai publish --version <버전>")


@main.command()
@click.option("--version", required=True, help="릴리스 버전 (예: 2026.07.15-1)")
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="pes 저장소 루트 (scripts/r2-put.py 위치, 기본: 현재 디렉토리의 상위)",
)
@click.option("--promote", is_flag=True, help="후보(candidate)가 아니라 즉시 발행(published)")
@click.option("--dry-run", is_flag=True, help="업로드 없이 매니페스트만 생성")
def publish(version: str, repo_root: Path | None, promote: bool, dry_run: bool) -> None:
    """(오너용) ONNX 모델과 토크나이저를 R2 에 올리고 릴리스를 등록한다.

    서버는 파일을 만지지 않는다 — 브라우저가 R2(cdn.getpes.com)에서 직접 내려받는다.
    R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY 환경변수가 필요하다.
    """
    cfg = _require_token()
    model_path = BUILD_DIR / "fai.int8.onnx"
    if not model_path.exists():
        raise click.ClickException("빌드 산출물이 없습니다. 먼저 `fai build` 를 실행하세요.")
    if not TOKENIZER_PATH.exists():
        raise click.ClickException("토크나이저가 없습니다. 먼저 `fai pull` 을 실행하세요.")

    from .checkpoint_io import load_checkpoint
    from .release import R2_PREFIX, make_manifest, register_release, upload_to_r2, write_manifest

    _, meta = load_checkpoint(CKPT_PATH) if CKPT_PATH.exists() else ({}, {})
    docs_count = sum(1 for _ in SNAPSHOT_PATH.open(encoding="utf-8")) if SNAPSHOT_PATH.exists() else 0

    manifest = make_manifest(
        version,
        model_path,
        TOKENIZER_PATH,
        docs_count=docs_count,
        train_loss=meta.get("train_loss"),
        val_loss=meta.get("val_loss"),
        step=int(meta.get("step", 0)),
    )
    manifest_path = write_manifest(manifest, BUILD_DIR)
    click.echo(f"[FAI] 매니페스트: {manifest_path}")
    click.echo(f"       모델 sha256: {manifest['model']['sha256'][:16]}… ({manifest['model']['bytes'] / 1e6:.1f}MB)")

    if dry_run:
        click.echo("[FAI] --dry-run — 업로드하지 않았습니다.")
        return

    root = repo_root or Path.cwd().parent
    click.echo(f"[FAI] R2 업로드 중 (cdn.getpes.com/{R2_PREFIX}/{version}/)…")
    upload_to_r2(model_path, f"{R2_PREFIX}/{version}/{model_path.name}", root)
    upload_to_r2(TOKENIZER_PATH, f"{R2_PREFIX}/{version}/{TOKENIZER_PATH.name}", root)
    upload_to_r2(manifest_path, f"{R2_PREFIX}/latest.json", root)

    status = "published" if promote else "candidate"
    result = register_release(cfg.server, cfg.token, manifest, status=status)
    click.echo(f"[FAI] 릴리스 등록 완료: {version} ({status}) id={result.get('id')}")
    if not promote:
        click.echo("[FAI] 평가 후 발행하려면: fai publish --version <버전> --promote")


if __name__ == "__main__":
    main()
