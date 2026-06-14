"""시퀀스 로더 — events.csv → LSTM 입력 텐서.

기능:
- customer_id 별 시간순 정렬
- event_type → 정수 인덱스 매핑 (0 = PAD)
- 시퀀스를 max_len 으로 통일 (truncating + left-padding)
- customer_id 매칭은 항상 str 키 (외부 데이터셋의 int id 자동 호환)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 8가지 event_type + PAD 토큰. 학습/추론 시 동일 인덱스 보장.
EVENT_TYPES = [
    "<PAD>",  # 0
    "page_view",  # 1
    "search",  # 2
    "add_to_cart",  # 3
    "remove_from_cart",  # 4
    "purchase",  # 5
    "coupon_use",  # 6
    "review",  # 7
    "cs_contact",  # 8
]
EVENT_TO_IDX = {name: idx for idx, name in enumerate(EVENT_TYPES)}
PAD_IDX = 0
VOCAB_SIZE = len(EVENT_TYPES)  # 9 (PAD 포함)


@dataclass
class SequenceData:
    """시퀀스 변환 결과.

    Attributes:
        sequences: (n_customers, max_len) int. 짧은 시퀀스는 앞쪽이 0(PAD).
        lengths: (n_customers,) 실제 시퀀스 길이.
        cid_to_row: customer_id(str) → sequences 행 인덱스.
                    키 dtype 은 항상 str 로 정규화 (외부 데이터셋의 int id 호환).
    """

    sequences: np.ndarray
    lengths: np.ndarray
    cid_to_row: dict[str, int]

    def select_by_cids(self, cids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """주어진 customer_id 순서대로 시퀀스 + 길이 추출.

        events.csv 에 없는 고객(이벤트 0건)은 전체 PAD 빈 시퀀스를 반환.
        KeyError 로 거부하지 않는 이유:
        - "ML/DL 동일 테스트셋 비교" — ML 도 이런 고객을 평가함
        - 시뮬레이터 small 모드에 약 3% (5,000명 중 163명) "유령 회원" 존재
        - PAD 시퀀스 입력은 LSTM 이 "행동 정보 없음" 으로 처리 → 라벨 누설 없음

        매칭은 항상 str 키 (Retailrocket 등 visitorid 가 int 인 외부 데이터셋 호환).
        """
        max_len = self.sequences.shape[1]

        out_seqs = np.zeros((len(cids), max_len), dtype=np.int64)
        out_lens = np.zeros(len(cids), dtype=np.int64)

        # int → str 자동 변환 (pd.read_csv 가 정수형 id 를 int64 로 추론하는 경우 대비)
        cids_str = cids.astype(str)

        missing_count = 0
        for i, cid in enumerate(cids_str):
            if cid in self.cid_to_row:
                row_idx = self.cid_to_row[cid]
                out_seqs[i] = self.sequences[row_idx]
                out_lens[i] = self.lengths[row_idx]
            else:
                missing_count += 1

        # logger.info 사용 (logger.debug 는 보통 출력 안 되어 처리 사실이 묻혀버림).
        if missing_count > 0:
            logger.info(
                "[SequenceData] 이벤트 없는 고객 %d명 → 빈 시퀀스(PAD) 처리 " "(ML/DL 동일 test set 보장)",
                missing_count,
            )

        return out_seqs, out_lens


def load_event_sequences(
    events_path: str | Path,
    max_len: int = 100,
    id_col: str = "customer_id",
    event_col: str = "event_type",
    date_col: str = "event_date",
) -> SequenceData:
    """events.csv → (n_customers, max_len) 시퀀스 텐서.

    - 최근 max_len 개 이벤트 보존 (이탈 직전 행동 중요).
    - left-padding: hidden state 가 0 → 첫 이벤트 → ... → 마지막 순으로 자연스럽게 누적.
    """
    if not isinstance(max_len, int) or max_len < 1:
        raise ValueError(
            f"[SeqLoader] max_len 은 1 이상의 정수. 받음: {max_len!r} " f"(type={type(max_len).__name__})"
        )

    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(
            f"[SeqLoader] events 파일 없음: {events_path}\n"
            "  → python src/main.py --mode simulate --sim-mode small 먼저 실행"
        )

    df = pd.read_csv(events_path, parse_dates=[date_col])
    logger.info("[SeqLoader] events 로드: %d 건", len(df))

    # customer_id 를 str 로 강제 변환 (feature_store 의 cid 와 dtype 일치).
    # Retailrocket 등 visitorid 가 int 인 외부 데이터셋에서 dict lookup 실패 방지.
    df[id_col] = df[id_col].astype(str)

    df["event_idx"] = df[event_col].map(EVENT_TO_IDX)
    n_unknown = int(df["event_idx"].isna().sum())
    if n_unknown:
        unknown_types = df.loc[df["event_idx"].isna(), event_col].unique()
        logger.warning(
            "[SeqLoader] 알 수 없는 event_type %d건 (PAD 처리): %s",
            n_unknown,
            list(unknown_types),
        )
        df["event_idx"] = df["event_idx"].fillna(PAD_IDX)
    df["event_idx"] = df["event_idx"].astype(int)

    # 시간순 정렬 필수
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    sequences_per_cid: dict[str, np.ndarray] = {}
    for cid, group in df.groupby(id_col, sort=False):
        events = group["event_idx"].tail(max_len).to_numpy(dtype=np.int64)
        sequences_per_cid[str(cid)] = events

    n_customers = len(sequences_per_cid)
    if n_customers == 0:
        raise ValueError("[SeqLoader] events.csv 에서 시퀀스 0건 추출됨.")

    sequences = np.zeros((n_customers, max_len), dtype=np.int64)
    lengths = np.zeros(n_customers, dtype=np.int64)
    cid_to_row: dict[str, int] = {}

    for row_idx, (cid, seq) in enumerate(sequences_per_cid.items()):
        seq_len = len(seq)
        sequences[row_idx, -seq_len:] = seq
        lengths[row_idx] = seq_len
        cid_to_row[cid] = row_idx

    logger.info(
        "[SeqLoader] sequences: n=%d, max_len=%d, avg_len=%.1f, padded=%.1f%%",
        n_customers,
        max_len,
        float(lengths.mean()),
        float((lengths < max_len).mean()) * 100,
    )

    return SequenceData(
        sequences=sequences,
        lengths=lengths,
        cid_to_row=cid_to_row,
    )
