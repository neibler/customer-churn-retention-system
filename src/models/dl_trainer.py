"""
DL 트레이너 — LSTM 시퀀스 모델 (배한솔, 태스크 2.14)

명세서 §5.5 요구사항 충족:
- 고객 행동 시퀀스 입력 LSTM
- 시퀀스 데이터 전처리 (패딩, 임베딩)  ← sequence_loader.py 가 담당
- Early Stopping
- ML 모델과 동일 테스트셋에서 비교 가능 (test_metrics 동일 포맷)
- DL 모델 파일/학습 로그/비교 리포트 저장 (joblib 대신 torch.save 사용)

설계 결정:
- Embedding → LSTM(2-layer, hidden=64) → Dropout → FC → Sigmoid
  Kumar & Kumar(2026) 의 이커머스 이탈 LSTM 권장 구조.
- CPU 환경 보장 (명세서 §7 제약): torch.device('cpu') 강제 가능.
  GPU 가 있으면 자동 사용.
- 클래스 불균형: SMOTE 가 시퀀스에 부적합 (k-NN 보간이 의미 없음) →
  BCEWithLogitsLoss 의 pos_weight 로 loss-level 처리.
- 모델 직렬화: torch.save() — pickle 의 S301 회피 + PyTorch 표준.
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── 서드파티 (지연 import 가능하지만 명시) ──────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# torch 는 무거운 import 라 함수 안에서 lazy load 도 고려했지만, 본 모듈은
# DL 전용이라 top-level import 가 자연스러움. 호출자가 dl_trainer 를 import
# 하지 않으면 torch import 도 안 일어남.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError(
        "PyTorch 가 필요합니다 (태스크 2.14, 명세서 §7 제약).\n"
        "  pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu\n"
        "  (CPU 전용 wheel. Docker 환경은 Dockerfile 이 별도 처리.)"
    ) from e

from src.models.sequence_loader import PAD_IDX, VOCAB_SIZE

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 안전 평가 헬퍼 — 단일 클래스 split 가드 (CodeRabbit Major 반영)
# ══════════════════════════════════════════════════════════════════════


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, split_name: str) -> float:
    """단일 클래스 split 에서도 안전한 ROC AUC 계산.

    sklearn 의 roc_auc_score 는 y_true 에 한 종류 라벨만 있으면
    'ValueError: Only one class present in y_true' 를 던진다.

    시나리오:
    - 작은 val split 에 우연히 양성 0건 (이탈률 20% × 50명 = 평균 10건이지만 분산 큼)
    - test_size 너무 작게 설정한 경우
    - stratify 옵션이 꺼진 채로 호출된 경우 (안전망)

    fallback: 0.5 (무작위 분류 기준점) + 경고 로그.
    이렇게 하면 학습 루프가 죽지 않고 다음 epoch 로 진행 가능.
    """
    if np.unique(y_true).size < 2:
        logger.warning(
            "[DL] %s split 에 단일 클래스만 존재 → AUC fallback 0.5 (label 분포 점검 필요)",
            split_name,
        )
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, split_name: str) -> float:
    """단일 클래스 split 에서도 안전한 PR AUC (average precision) 계산.

    average_precision_score 는 y_true 에 양성이 0건이면 NaN 또는 정의 불가.
    fallback: 0.0 (precision 측정 불가) + 경고 로그.
    """
    if np.unique(y_true).size < 2:
        logger.warning("[DL] %s split 에 단일 클래스만 존재 → PR-AUC fallback 0.0", split_name)
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ══════════════════════════════════════════════════════════════════════
# LSTM 모델 정의
# ══════════════════════════════════════════════════════════════════════


class ChurnLSTM(nn.Module):
    """이탈 예측용 LSTM 모델.

    구조:
        Embedding(vocab=9, dim=16, padding_idx=0)
            → LSTM(input=16, hidden=64, layers=2, dropout=0.2)
            → 마지막 hidden state 추출
            → Dropout(0.3)
            → Linear(64, 1)
            → (sigmoid 는 BCEWithLogitsLoss 가 내부 처리하므로 생략)

    파라미터 수 추정:
        Embedding:  9 x 16 = 144
        LSTM L1:    4 x (16 + 64 + 1) x 64 = 20,736
        LSTM L2:    4 x (64 + 64 + 1) x 64 = 33,024
        Linear:     64 + 1 = 65
        총합:       약 54k params (CPU 학습 충분히 빠름)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        n_layers: int = 2,
        lstm_dropout: float = 0.2,
        fc_dropout: float = 0.3,
    ):
        super().__init__()
        # padding_idx=0: PAD 토큰의 임베딩을 0 으로 고정 + gradient 차단
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX,
        )

        # batch_first=True: 입력 shape (batch, seq_len, embed_dim) 으로 직관적
        # dropout 은 layer 사이에만 적용 (n_layers > 1 일 때만 작동)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: shape (batch, seq_len) int64. event_idx 시퀀스.

        Returns:
            logits: shape (batch,) float. sigmoid 전 raw score.
                    BCEWithLogitsLoss 가 sigmoid 를 내부에서 처리.
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # LSTM 출력: (batch, seq_len, hidden_dim), (h_n, c_n)
        # 마지막 시점의 hidden state (h_n[-1]) 를 분류용 표현으로 사용.
        # h_n shape: (n_layers, batch, hidden_dim) → 마지막 layer 만 추출
        _output, (h_n, _c_n) = self.lstm(embedded)
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # FC 분류기
        dropped = self.fc_dropout(last_hidden)
        logits = self.fc(dropped).squeeze(-1)  # (batch,)
        return logits


# ══════════════════════════════════════════════════════════════════════
# 학습 결과 컨테이너 (ml_trainer 의 CVResult 와 유사 패턴)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class DLTrainResult:
    """DL 학습 결과. ml_trainer.CVResult 와 호환되는 인터페이스."""

    test_metrics: dict[str, float] = field(default_factory=dict)
    test_proba: np.ndarray | None = None
    best_val_auc: float = 0.0
    best_epoch: int = -1
    epochs_trained: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    final_model: Any = None

    def summary(self) -> str:
        """리포트용 요약 문자열."""
        lines = [
            "=== LSTM Result ===",
            f"  Epochs trained: {self.epochs_trained} (best at epoch {self.best_epoch})",
            f"  Best val AUC  : {self.best_val_auc:.4f}",
            "  Test metrics:",
        ]
        for k, v in self.test_metrics.items():
            lines.append(f"    {k:12s} = {v:.4f}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 학습 함수
# ══════════════════════════════════════════════════════════════════════


def _evaluate(
    model: ChurnLSTM,
    loader: "DataLoader",
    device: "torch.device",
) -> tuple[np.ndarray, np.ndarray]:
    """추론 전용 헬퍼. (proba, y_true) 반환."""
    model.eval()
    all_proba = []
    all_y = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            proba = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_y.append(y_batch.numpy())
    return np.concatenate(all_proba), np.concatenate(all_y)


def train_lstm(
    seq_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    y_val: np.ndarray,
    seq_test: np.ndarray,
    y_test: np.ndarray,
    *,
    embed_dim: int = 16,
    hidden_dim: int = 64,
    n_layers: int = 2,
    lstm_dropout: float = 0.2,
    fc_dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 30,
    early_stopping_patience: int = 3,
    pos_weight_auto: bool = True,
    device: str = "auto",
    random_state: int = 42,
    log_file: str | Path | None = None,
) -> DLTrainResult:
    """LSTM 학습 + Early Stopping + Test 평가.

    Args:
        seq_*: shape (n, max_len) numpy int64. sequence_loader 출력.
        y_*: shape (n,) numpy int. 이탈 라벨.
        pos_weight_auto: True 면 train 셋의 (n_neg / n_pos) 비율로
            BCEWithLogitsLoss 의 pos_weight 자동 설정 (클래스 불균형 처리).
        device: "auto" / "cpu" / "cuda".
        log_file: epoch 별 학습 로그를 별도 파일로 저장.

    Returns:
        DLTrainResult — test_metrics, best epoch, history, model 포함.
    """
    # ── 재현성 시드 ──────────────────────────────────────────
    # ml_trainer 와 동일한 random_state 패턴.
    # cuDNN 비결정성은 CPU 환경에선 영향 없음.
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # ── 디바이스 결정 ────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(
        "[DL] device=%s, train=%d, val=%d, test=%d",
        device,
        len(y_train),
        len(y_val),
        len(y_test),
    )

    # ── DataLoader 구성 ─────────────────────────────────────
    def _make_loader(seq: np.ndarray, y: np.ndarray, shuffle: bool) -> "DataLoader":
        ds = TensorDataset(
            torch.from_numpy(seq).long(),
            torch.from_numpy(y).float(),
        )
        # num_workers=0: CPU 환경 + 작은 데이터(5k~20k) 에서는 worker 오버헤드 ↑
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(seq_train, y_train, shuffle=True)
    val_loader = _make_loader(seq_val, y_val, shuffle=False)
    test_loader = _make_loader(seq_test, y_test, shuffle=False)

    # ── 모델 + Optimizer + Loss ─────────────────────────────
    model = ChurnLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lstm_dropout=lstm_dropout,
        fc_dropout=fc_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 클래스 불균형 처리: pos_weight = n_neg / n_pos.
    # SMOTE 가 시퀀스에 부적합하므로 loss-level 처리 (명세서 §5.4.2 권장 옵션
    # 중 class_weight 와 동등 — 다만 BCEWithLogitsLoss 의 pos_weight 형태).
    if np.unique(y_train).size < 2:
        raise ValueError("[DL] train split must contain both classes")

    if pos_weight_auto:
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        logger.info(
            "[DL] pos_weight=%.3f (n_neg=%.0f / n_pos=%.0f)",
            pos_weight.item(),
            n_neg,
            n_pos,
        )
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Early Stopping 상태 ──────────────────────────────────
    best_val_auc = -np.inf
    best_epoch = -1
    best_state_dict: dict | None = None
    patience_counter = 0
    history: list[dict[str, float]] = []

    # 별도 학습 로그 파일 (명세서 §5.5.6 "DL 모델 파일/학습 로그/ML 대비 비교
    # 리포트를 저장하고, 모델 선택 근거를 문서화해야 한다")
    log_fp = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fp = Path(log_file).open("w", encoding="utf-8")
        log_fp.write("epoch\ttrain_loss\tval_auc\tval_pr_auc\tval_f1\n")

    # ── 학습 루프 ────────────────────────────────────────────
    try:
        for epoch in range(1, max_epochs + 1):
            # train
            model.train()
            train_loss_sum = 0.0
            n_samples = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(y_batch)
                n_samples += len(y_batch)
            train_loss = train_loss_sum / n_samples

            # val 평가 — 단일 클래스 split 가드 사용 (CodeRabbit Major)
            val_proba, val_y = _evaluate(model, val_loader, device)
            val_pred = (val_proba >= 0.5).astype(int)
            val_auc = _safe_roc_auc(val_y, val_proba, "val")
            val_pr_auc = _safe_pr_auc(val_y, val_proba, "val")
            val_f1 = float(f1_score(val_y, val_pred, zero_division=0))

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_auc": val_auc,
                    "val_pr_auc": val_pr_auc,
                    "val_f1": val_f1,
                }
            )
            logger.info(
                "[DL] epoch %d: train_loss=%.4f val_AUC=%.4f val_PR-AUC=%.4f val_F1=%.4f",
                epoch,
                train_loss,
                val_auc,
                val_pr_auc,
                val_f1,
            )
            if log_fp:
                log_fp.write(f"{epoch}\t{train_loss:.4f}\t{val_auc:.4f}\t{val_pr_auc:.4f}\t{val_f1:.4f}\n")
                log_fp.flush()

            # Early Stopping: val AUC 가 개선되면 체크포인트 저장, 아니면 카운터 증가
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                # state_dict 만 보관 (deepcopy 보다 빠르고 메모리 적음)
                best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        "[DL] Early stopping at epoch %d (patience=%d, best epoch=%d)",
                        epoch,
                        early_stopping_patience,
                        best_epoch,
                    )
                    break
    finally:
        if log_fp:
            log_fp.close()

    # ── Best 가중치 복원 + Test 평가 ─────────────────────────
    if best_state_dict is None:
        raise RuntimeError("[DL] 학습 중 best state 미저장 — 코드 버그 가능성")
    model.load_state_dict(best_state_dict)

    test_proba, test_y = _evaluate(model, test_loader, device)
    test_pred = (test_proba >= 0.5).astype(int)

    # Test 평가도 안전 가드 적용 (test 도 우연히 단일 클래스 가능성 — drop_last 등)
    test_metrics = {
        "auc": _safe_roc_auc(test_y, test_proba, "test"),
        "pr_auc": _safe_pr_auc(test_y, test_proba, "test"),
        "f1": float(f1_score(test_y, test_pred, zero_division=0)),
        "precision": float(precision_score(test_y, test_pred, zero_division=0)),
        "recall": float(recall_score(test_y, test_pred, zero_division=0)),
    }

    return DLTrainResult(
        test_metrics=test_metrics,
        test_proba=test_proba,
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        epochs_trained=history[-1]["epoch"] if history else 0,
        history=history,
        final_model=model,
    )


# ══════════════════════════════════════════════════════════════════════
# 모델 영속화 (명세서 §5.5.6 "DL 모델 파일 저장")
# ══════════════════════════════════════════════════════════════════════


def save_dl_model(model: ChurnLSTM, path: str | Path) -> None:
    """LSTM 모델 가중치 + 구조 정보 저장.

    torch.save(state_dict) 만 하면 추론 시 모델 구조를 별도로 알아야 함 →
    state_dict + 하이퍼파라미터를 dict 로 묶어 저장.

    명세서 §5.5.6 "DL 모델 파일/학습 로그/ML 대비 비교 리포트를 저장" 충족.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 모델 구조 재현에 필요한 하이퍼파라미터도 함께 저장.
    # 추론 시 ChurnLSTM(**checkpoint['hparams']) 로 재구성 가능.
    checkpoint = {
        "state_dict": model.state_dict(),
        "hparams": {
            "vocab_size": model.embedding.num_embeddings,
            "embed_dim": model.embedding.embedding_dim,
            "hidden_dim": model.lstm.hidden_size,
            "n_layers": model.lstm.num_layers,
        },
    }
    torch.save(checkpoint, p)
    logger.info("[DL] saved → %s", p)


def load_dl_model(path: str | Path) -> ChurnLSTM:
    """LSTM 모델 로드. save_dl_model 의 짝.

    보안 주의: torch.load 도 내부적으로 pickle 사용. 신뢰할 수 있는 출처의
    .pt 파일만 로드할 것 (ml_trainer.load_model 과 동일한 가드 정신).
    """
    # weights_only=True 는 PyTorch 2.0+ 의 보안 옵션이지만 state_dict 만 로드
    # 가능하고 hparams dict 를 못 읽어서 weights_only=False 유지.
    # 신뢰할 수 있는 자체 학습 산출물만 로드한다는 가정.
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model = ChurnLSTM(**checkpoint["hparams"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


def save_dl_metrics(result: DLTrainResult, path: str | Path) -> None:
    """DL 학습 결과를 JSON 으로 저장 (ML vs DL 비교용)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "test_metrics": result.test_metrics,
        "best_val_auc": result.best_val_auc,
        "best_epoch": result.best_epoch,
        "epochs_trained": result.epochs_trained,
        "history": result.history,
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("[DL] metrics saved → %s", p)
