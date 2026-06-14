"""YAML 설정 파일 로더. 하드코딩 금지(G3) 강제용."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "config/model_config.yaml") -> dict[str, Any]:
    """YAML 설정을 dict 로 로드.

    설정 파일이 없으면 silent default 로 빠지지 않고 명시적으로 FileNotFoundError 를
    던진다 — "어떤 설정으로 학습됐는지 모르는" 상태가 가장 위험하기 때문.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"[Config] 설정 파일 없음: {path}")

    # safe_load: 임의 파이썬 객체 역직렬화 차단 (yaml.load 는 RCE 위험).
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"[Config] {path} 파싱 실패: dict 가 아님")

    return cfg
