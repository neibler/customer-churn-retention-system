"""설정 파일 로더 (배한솔). 하드코딩 금지 원칙(G3)."""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import (
    annotations,
)  # 파이썬 3.9 이하에서도 PEP 604 union(|) 문법 사용 가능하게

from pathlib import Path  # os.path 보다 안전하고 가독성 좋은 경로 처리
from typing import Any  # YAML 값은 어떤 타입이든 올 수 있으므로 Any 로 받음

# ── 서드파티 ──────────────────────────────────────────────────
import yaml  # PyYAML. requirements.txt 에 pyyaml 추가 필요


def load_config(config_path: str | Path = "config/model_config.yaml") -> dict[str, Any]:
    """YAML 설정 파일을 dict 로 로드한다.

    하드코딩 금지 원칙(G3) 을 강제하기 위해, 모든 하이퍼파라미터/경로/옵션은
    이 함수를 통해서만 가져오도록 통일한다. 설정 파일이 존재하지 않으면
    실수 방지를 위해 명시적으로 FileNotFoundError 를 던진다 (silent default 금지).

    Args:
        config_path: 프로젝트 루트 기준 상대 경로 또는 절대 경로.
            기본값은 팀 합의된 위치 `config/model_config.yaml`.

    Returns:
        파싱된 dict. 최상위 키 예: paths, data, xgboost, lightgbm, optuna, ...

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: YAML 이 dict 가 아닐 때 (예: 리스트만 있거나 빈 파일).
    """
    # str 로 받아도 Path 로 정규화 → 이후 .exists(), .open() 등 사용 가능
    path = Path(config_path)

    # 명시적으로 파일 부재를 알려줌. Default config 로 fallback 하지 않는 이유:
    # "어떤 설정으로 학습됐는지 모르는" 상태가 가장 위험하기 때문.
    if not path.exists():
        raise FileNotFoundError(f"[Config] 설정 파일 없음: {path}")

    # 인코딩을 명시적으로 utf-8 로 (Windows 환경에서 cp949 기본값 문제 회피)
    with path.open("r", encoding="utf-8") as f:
        # safe_load 는 임의 파이썬 객체 역직렬화를 막아 보안적으로 안전.
        # yaml.load 는 RCE 위험이 있어 절대 사용하지 않는다.
        cfg = yaml.safe_load(f)

    # YAML 빈 파일 → None, 또는 리스트만 있는 경우 → list 등 잘못된 구조 방어.
    if not isinstance(cfg, dict):
        raise ValueError(f"[Config] {path} 파싱 실패: dict 가 아님")

    return cfg
