"""DL 트레이너 — LSTM 시퀀스 모델.

구조: Embedding → LSTM(2-layer, hidden=64) → Dropout → FC → BCEWithLogitsLoss
- 클래스 불균형: pos_weight (SMOTE 가 시퀀스에 부적합).
- Early Stopping: val AUC 기준 patience=3, best state_dict 복원.
- 직렬화: torch.save (pickle S301 회피 + PyTorch 표준).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError("PyTorch 필요: pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu") from e

from src.models.sequence_loader import PAD_IDX, VOCAB_SIZE

logger = logging.getLogger(__name__)


# ── 안전 평가 헬퍼 — 단일 클래스 split 가드 ────────────────────────────
# val/test split 에 우연히 양성 0건이면 roc_auc_score 가 ValueError 를 던져
# 학습 루프가 중단됨. fallback 으로 학습 진행은 유지하고 경고만 남긴다.
# (train split 에서는 fail-fast — 학습 자체가 무의미해지므로)
def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, split_name: str) -> float:
    if np.unique(y_true).size < 2:
        logger.warning("[DL] %s split 단일 클래스 → AUC fallback 0.5", split_name)
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, split_name: str) -> float:
    if np.unique(y_true).size < 2:
        logger.warning("[DL] %s split 단일 클래스 → PR-AUC fallback 0.0", split_name)
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ── LSTM 모델 정의 ────────────────────────────────────────────
class ChurnLSTM(nn.Module):
    """LSTM 분류 모델.

    Embedding(9, 16, padding_idx=0) → LSTM(16→64, 2-layer, dropout=0.2)
        → Dropout(0.3) → Linear(64, 1) → BCEWithLogitsLoss 내부 sigmoid.
    파라미터 약 54k (CPU 학습 충분).
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
        # padding_idx=0: PAD 토큰 임베딩을 0 으로 고정 + gradient 차단
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
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
        """x: (batch, seq_len) int64 → logits: (batch,) float."""
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        # h_n: (n_layers, batch, hidden) → 마지막 layer 만 사용
        last_hidden = h_n[-1]
        return self.fc(self.fc_dropout(last_hidden)).squeeze(-1)


# ── 학습 결과 컨테이너 ────────────────────────────────────────────
@dataclass
class DLTrainResult:
    test_metrics: dict[str, float] = field(default_factory=dict)
    test_proba: np.ndarray | None = None
    best_val_auc: float = 0.0
    best_epoch: int = -1
    epochs_trained: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    final_model: Any = None

    def summary(self) -> str:
        lines = [
            "=== LSTM Result ===",
            f"  Epochs trained: {self.epochs_trained} (best at epoch {self.best_epoch})",
            f"  Best val AUC  : {self.best_val_auc:.4f}",
            "  Test metrics:",
        ]
        for k, v in self.test_metrics.items():
            lines.append(f"    {k:12s} = {v:.4f}")
        return "\n".join(lines)


# ── 학습 함수 ─────────────────────────────────────────────────
def _evaluate(model: ChurnLSTM, loader: "DataLoader", device: "torch.device") -> tuple[np.ndarray, np.ndarray]:
    """추론 → (proba, y_true)."""
    model.eval()
    all_proba, all_y = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
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
    """LSTM 학습 + Early Stopping + Test 평가."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("[DL] device=%s, train=%d, val=%d, test=%d", device, len(y_train), len(y_val), len(y_test))

    def _make_loader(seq, y, shuffle):
        ds = TensorDataset(torch.from_numpy(seq).long(), torch.from_numpy(y).float())
        # num_workers=0: CPU + 작은 데이터(5k~20k)에서 worker 오버헤드 ↑
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(seq_train, y_train, shuffle=True)
    val_loader = _make_loader(seq_val, y_val, shuffle=False)
    test_loader = _make_loader(seq_test, y_test, shuffle=False)

    model = ChurnLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lstm_dropout=lstm_dropout,
        fc_dropout=fc_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train 단일 클래스는 fail-fast — pos_weight 계산이 무의미해지고 untrained
    # 모델이 silent 하게 반환되는 위험 시나리오라 학습 자체를 진행하면 안 됨.
    if np.unique(y_train).size < 2:
        raise ValueError(
            f"[DL] train split 단일 클래스 (labels={np.unique(y_train).tolist()}). " "data split / stratify 점검 필요."
        )

    # SMOTE 부적합 (k-NN 보간이 시퀀스에 의미 없음) → pos_weight 로 loss-level 처리.
    if pos_weight_auto:
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        logger.info("[DL] pos_weight=%.3f (n_neg=%.0f / n_pos=%.0f)", pos_weight.item(), n_neg, n_pos)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -np.inf
    best_epoch = -1
    best_state_dict: dict | None = None
    patience_counter = 0
    history: list[dict[str, float]] = []

    log_fp = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fp = Path(log_file).open("w", encoding="utf-8")
        log_fp.write("epoch\ttrain_loss\tval_auc\tval_pr_auc\tval_f1\n")

    try:
        for epoch in range(1, max_epochs + 1):
            # train
            model.train()
            train_loss_sum, n_samples = 0.0, 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * len(y_batch)
                n_samples += len(y_batch)
            train_loss = train_loss_sum / n_samples

            # val
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

            # Early Stopping: val AUC 개선 시 체크포인트 저장 (deepcopy 회피용 state_dict clone)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        "[DL] Early stopping at epoch %d (patience=%d, best=%d)",
                        epoch,
                        early_stopping_patience,
                        best_epoch,
                    )
                    break
    finally:
        if log_fp:
            log_fp.close()

    if best_state_dict is None:
        raise RuntimeError("[DL] best state 미저장 — 코드 버그")
    model.load_state_dict(best_state_dict)

    test_proba, test_y = _evaluate(model, test_loader, device)
    test_pred = (test_proba >= 0.5).astype(int)

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


# ── 모델 영속화 ────────────────────────────────────────────────
def save_dl_model(model: ChurnLSTM, path: str | Path) -> None:
    """LSTM state_dict + 하이퍼파라미터를 함께 저장.

    state_dict 만으로는 추론 시 모델 구조를 별도로 알아야 함. hparams 동봉 →
    load_dl_model 에서 ChurnLSTM(**hparams) 로 재구성 가능.
    체크포인트는 dict / OrderedDict[str, Tensor] / 원시 int 만으로 구성 →
    load 시 weights_only=True 와 호환.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
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
    """LSTM 로드. weights_only=True (PyTorch 2.0+) 로 임의 클래스 pickle 실행 차단."""
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = ChurnLSTM(**checkpoint["hparams"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def save_dl_metrics(result: DLTrainResult, path: str | Path) -> None:
    """학습 결과를 JSON 으로 저장 (ML vs DL 비교용)."""
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
