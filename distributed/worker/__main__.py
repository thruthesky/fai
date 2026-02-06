# distributed/worker/__main__.py
# ============================================================================
# python -m distributed.worker 실행 진입점
# ============================================================================
# uv run python -m distributed.worker --name "워커 이름" --server http://localhost:8000

from .cli import main

main()
