# distributed/worker/cli.py
# ============================================================================
# 워커 CLI 진입점 (Click 기반)
# ============================================================================
# python -m distributed.worker 명령으로 워커를 실행합니다.
#
# 【사용법】
# uv run python -m distributed.worker \
#     --name "철수의 맥북" \
#     --server http://localhost:8000 \
#     --experiment 1
#
# 【학습 루프】
# 1. 서버에 워커 등록 (register)
# 2. 작업 요청 (request_task)
# 3. 체크포인트 다운로드 → 로컬 N 스텝 학습
# 4. 학습 완료 보고 + 가중치 업로드 (complete_task)
# 5. 2~4 반복 (무한 or max_rounds 만큼)
# 6. 종료 시 이탈 통보 (leave)

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import uuid
from functools import partial
from pathlib import Path

import click

from ..common.model import GPTConfig
from .checkpoint_io import CheckpointIO
from .client import CoordinatorClient
from .config import WorkerConfig
from .device_manager import detect_device
from .trainer import LocalTrainer

logger = logging.getLogger(__name__)


# ============================================================================
# 메인 학습 루프
# ============================================================================
async def run_worker(config: WorkerConfig) -> None:
    """
    워커 메인 루프

    【흐름】
    1. 디바이스 감지 (GPU/CPU)
    2. 서버 등록
    3. 반복: 작업 요청 → 체크포인트 다운 → 학습 → 업로드
    4. 종료 시 이탈 통보
    """
    # ────────────────────────────────────────────────────────────────
    # 디바이스 감지
    # ────────────────────────────────────────────────────────────────
    device_info = detect_device(preferred=config.device)
    logger.info(
        f"디바이스 감지: {device_info.device_type} "
        f"({device_info.device_name}, "
        f"GPU={device_info.gpu_memory_mb}MB, "
        f"RAM={device_info.ram_mb}MB, "
        f"CPU={device_info.cpu_cores}코어)"
    )

    # ────────────────────────────────────────────────────────────────
    # 캐시 및 상태 관리
    # ────────────────────────────────────────────────────────────────
    cache_path = config.cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    ckpt_io = CheckpointIO(cache_dir=cache_path)

    # 이전 등록 정보 복원 시도
    worker_uid = config.worker_uid
    if worker_uid is None and config.state_file.exists():
        try:
            state = json.loads(config.state_file.read_text())
            worker_uid = uuid.UUID(state["worker_uid"])
            logger.info(f"이전 등록 정보 복원: {worker_uid}")
        except Exception:
            worker_uid = None

    # ────────────────────────────────────────────────────────────────
    # 서버 연결 및 학습 루프
    # ────────────────────────────────────────────────────────────────
    async with CoordinatorClient(config.server_url) as client:
        # 워커 등록
        if worker_uid is None:
            reg_result = await client.register(
                name=config.name,
                device_type=device_info.device_type,
                device_name=device_info.device_name,
                gpu_memory_mb=device_info.gpu_memory_mb,
                ram_mb=device_info.ram_mb,
                cpu_cores=device_info.cpu_cores,
            )
            worker_uid = uuid.UUID(reg_result["worker_uid"])

            # 서버 추천 설정 적용
            if config.batch_size is None:
                config.batch_size = reg_result.get("recommended_batch_size", 16)
            if config.local_steps is None:
                config.local_steps = reg_result.get("recommended_local_steps", 100)

            # 등록 정보 저장
            config.state_file.parent.mkdir(parents=True, exist_ok=True)
            config.state_file.write_text(json.dumps({
                "worker_uid": str(worker_uid),
            }))
            logger.info(f"워커 등록 완료: uid={worker_uid}")
        else:
            # 기존 등록으로 heartbeat 전송
            await client.heartbeat(worker_uid, status="online")
            logger.info(f"기존 등록으로 재접속: uid={worker_uid}")

        # 기본값 설정
        batch_size = config.batch_size or 16
        local_steps = config.local_steps or 100
        learning_rate = config.learning_rate

        # 트레이너 초기화
        trainer = LocalTrainer(
            data_dir=config.data_dir,
            device=device_info.torch_device,
        )

        # Heartbeat 콜백 생성
        async def heartbeat_callback():
            await client.heartbeat(worker_uid, status="training")

        # ────────────────────────────────────────────────────────────
        # 학습 라운드 반복
        # ────────────────────────────────────────────────────────────
        round_num = 0
        running = True

        # 종료 시그널 처리 (Ctrl+C)
        def signal_handler(sig, frame):
            nonlocal running
            logger.info("종료 시그널 수신 — 현재 라운드 완료 후 종료합니다...")
            running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while running:
            round_num += 1

            # 최대 라운드 체크
            if config.max_rounds and round_num > config.max_rounds:
                logger.info(f"최대 라운드 도달 ({config.max_rounds}), 종료합니다.")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"라운드 {round_num} 시작")
            logger.info(f"{'='*60}")

            try:
                # 1. 작업 요청
                task = await client.request_task(
                    worker_uid=worker_uid,
                    experiment_id=config.experiment_id,
                )
                task_id = task["task_id"]
                checkpoint_url = task.get("checkpoint_url")

                logger.info(
                    f"작업 수신: task_id={task_id}, "
                    f"local_steps={task.get('local_steps', local_steps)}"
                )

                # 서버 지정 스텝 수 사용 (있으면)
                steps = task.get("local_steps", local_steps)

                # 2. 체크포인트 다운로드 및 모델 로드
                # GPTConfig는 서버에서 받거나 기본값 사용
                gpt_config_dict = task.get("model_config", {})
                if gpt_config_dict:
                    gpt_config = GPTConfig.from_dict(gpt_config_dict)
                else:
                    gpt_config = GPTConfig()

                if checkpoint_url:
                    model = await ckpt_io.download_and_load(
                        client=client,
                        checkpoint_url=checkpoint_url,
                        config=gpt_config,
                        device=device_info.torch_device,
                    )
                else:
                    # 체크포인트 없으면 새 모델 (첫 라운드)
                    from ..common.model import GPT
                    model = GPT(gpt_config).to(device_info.torch_device)
                    logger.info("체크포인트 없음 — 새 모델로 시작합니다.")

                # 3. 로컬 학습
                result = await trainer.train(
                    model=model,
                    local_steps=steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    heartbeat_fn=heartbeat_callback,
                )

                # 4. 가중치 저장 및 업로드
                weights_path = ckpt_io.save_trained_weights(model, task_id)

                await client.complete_task(
                    task_id=task_id,
                    worker_uid=worker_uid,
                    steps_trained=result.steps_trained,
                    local_train_loss=result.train_loss,
                    local_val_loss=result.val_loss,
                    training_duration_s=result.duration_s,
                    device_type=device_info.device_type,
                    batch_size_used=result.batch_size_used,
                    learning_rate_used=result.learning_rate_used,
                    weights_path=weights_path,
                )
                logger.info(f"라운드 {round_num} 완료 — 가중치 업로드 성공")

                # 5. 임시 파일 정리
                ckpt_io.cleanup_task_weights(task_id)
                ckpt_io.cleanup_old_checkpoints(keep_latest=3)

                # 모델 메모리 해제
                del model
                if device_info.device_type == "cuda":
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                logger.info("키보드 인터럽트 — 종료합니다.")
                break
            except Exception as e:
                logger.error(f"라운드 {round_num} 실패: {e}", exc_info=True)
                # 잠시 대기 후 재시도
                logger.info("30초 후 재시도합니다...")
                await asyncio.sleep(30)
                continue

        # ────────────────────────────────────────────────────────────
        # 종료 처리
        # ────────────────────────────────────────────────────────────
        logger.info("워커 종료 중...")
        await client.leave(worker_uid)
        logger.info("서버에 이탈 통보 완료. 워커를 종료합니다.")


# ============================================================================
# Click CLI
# ============================================================================
@click.command("worker")
@click.option("--name", default="FAI 워커", help="워커 이름 (예: '철수의 맥북')")
@click.option("--server", default="http://localhost:8000", help="Coordinator 서버 URL")
@click.option("--experiment", "experiment_id", default=1, type=int, help="참여할 실험 ID")
@click.option("--device", default=None, type=click.Choice(["cuda", "mps", "cpu"]), help="디바이스 (기본: 자동 감지)")
@click.option("--batch-size", default=None, type=int, help="배치 크기 (기본: 서버 추천)")
@click.option("--local-steps", default=None, type=int, help="라운드당 학습 스텝 (기본: 서버 추천)")
@click.option("--lr", "learning_rate", default=3e-4, type=float, help="학습률")
@click.option("--data-dir", default="data", help="학습 데이터 디렉토리")
@click.option("--max-rounds", default=None, type=int, help="최대 학습 라운드 (기본: 무한)")
@click.option("--verbose", is_flag=True, help="상세 로그 출력")
def main(
    name: str,
    server: str,
    experiment_id: int,
    device: str | None,
    batch_size: int | None,
    local_steps: int | None,
    learning_rate: float,
    data_dir: str,
    max_rounds: int | None,
    verbose: bool,
) -> None:
    """
    FAI 분산 학습 워커를 실행합니다.

    \b
    【사용 예시】
    uv run python -m distributed.worker --name "철수의 맥북" --server http://coordinator:8000

    \b
    【워커 동작】
    1. 서버에 등록 → 작업 요청 → 체크포인트 다운로드
    2. 로컬 데이터로 N 스텝 학습
    3. 학습 결과 + 가중치 업로드
    4. 1~3 반복 (Ctrl+C로 종료)
    """
    # 로깅 설정
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 배너 출력
    click.echo("=" * 60)
    click.echo("  FAI 분산 학습 워커")
    click.echo("=" * 60)
    click.echo(f"  이름: {name}")
    click.echo(f"  서버: {server}")
    click.echo(f"  실험 ID: {experiment_id}")
    click.echo(f"  디바이스: {device or '자동 감지'}")
    click.echo(f"  최대 라운드: {max_rounds or '무한'}")
    click.echo("=" * 60)

    # 설정 생성
    config = WorkerConfig(
        name=name,
        server_url=server,
        experiment_id=experiment_id,
        device=device,
        batch_size=batch_size,
        local_steps=local_steps,
        learning_rate=learning_rate,
        data_dir=data_dir,
        max_rounds=max_rounds,
        verbose=verbose,
    )

    # 비동기 메인 루프 실행
    asyncio.run(run_worker(config))


if __name__ == "__main__":
    main()
