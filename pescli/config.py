# pescli 설정 — ~/.fai/config.json 에 서버 주소와 API 토큰을 보관한다.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".fai"
CONFIG_PATH = CONFIG_DIR / "config.json"

# 작업 디렉토리 (데이터·체크포인트·빌드 산출물)
WORK_DIR = CONFIG_DIR / "work"
DATA_DIR = WORK_DIR / "data"
CKPT_DIR = WORK_DIR / "checkpoints"
BUILD_DIR = WORK_DIR / "build"

DEFAULT_SERVER = "https://getpes.com"

# 한국어 중심 재학습 기준 하이퍼파라미터 (docs/pai-analysis.md 결정 사항)
VOCAB_SIZE = 32000
BLOCK_SIZE = 256
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


@dataclass
class CliConfig:
    server: str = DEFAULT_SERVER
    token: str = ""

    @classmethod
    def load(cls) -> "CliConfig":
        if CONFIG_PATH.exists():
            d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return cls(server=d.get("server", DEFAULT_SERVER), token=d.get("token", ""))
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps({"server": self.server, "token": self.token}, indent=2),
            encoding="utf-8",
        )
        CONFIG_PATH.chmod(0o600)  # 토큰 보호


def ensure_dirs() -> None:
    for d in (DATA_DIR, CKPT_DIR, BUILD_DIR):
        d.mkdir(parents=True, exist_ok=True)
