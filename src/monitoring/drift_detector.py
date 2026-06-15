import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path
from datetime import datetime

class DriftDetector:
    """
    피처 분포 변화(Data Drift)를 탐지하는 클래스.
    PSI(Population Stability Index) 및 KS-test(Kolmogorov-Smirnov test)를 사용함.
    """
    
    def __init__(self, threshold_psi=0.2, threshold_ks=0.05, threshold_performance_drop=0.05):
        self.threshold_psi = threshold_psi
        self.threshold_ks = threshold_ks
        self.threshold_performance_drop = threshold_performance_drop
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "performance": {},
            "alerts": []
        }

    def calculate_psi(self, expected, actual, buckets=10):
        """
        PSI 계산 로직.
        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        """
        def get_probs(data, bins):
            counts, _ = np.histogram(data, bins=bins)
            probs = counts / len(data)
            # 0 확률 방지 (Laplace smoothing 유사 처리)
            probs = np.where(probs == 0, 0.0001, probs)
            return probs

        # Binning 범위를 expected 기준으로 설정
        _, bin_edges = np.histogram(expected, bins=buckets)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        expected_probs = get_probs(expected, bin_edges)
        actual_probs = get_probs(actual, bin_edges)
        
        psi_value = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
        return float(psi_value)

    def calculate_ks(self, expected, actual):
        """
        KS-test 계산 로직.
        두 분포가 동일한 모집단에서 나왔는지 검정. p-value가 작을수록 분포가 다름을 의미.
        """
        result = stats.ks_2samp(expected, actual)
        return float(result.statistic), float(result.pvalue)

    def run_monitoring(self, reference_df, current_df, feature_cols):
        """
        지정된 피처들에 대해 모니터링 수행
        """
        self.report["metrics"] = {}
        # alerts는 초기화하지 않고 누적 (성능 알림이 먼저 들어올 수 있음)

        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
                
            ref_data = reference_df[col].dropna()
            curr_data = current_df[col].dropna()
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue

            # PSI 계산
            psi = self.calculate_psi(ref_data, curr_data)
            
            # KS 계산
            ks_stat, p_val = self.calculate_ks(ref_data, curr_data)
            
            self.report["metrics"][col] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": p_val
            }
            
            # Alert 체크
            if psi > self.threshold_psi:
                self.report["alerts"].append({
                    "feature": col,
                    "type": "PSI",
                    "value": psi,
                    "threshold": self.threshold_psi,
                    "message": f"High drift detected in feature '{col}' (PSI: {psi:.4f})"
                })
            
            if p_val < self.threshold_ks:
                self.report["alerts"].append({
                    "feature": col,
                    "type": "KS",
                    "value": p_val,
                    "threshold": self.threshold_ks,
                    "message": f"Significant distribution change in feature '{col}' (KS p-value: {p_val:.4f})"
                })

        return self.report

    def calculate_performance_metrics(self, y_true, y_proba, threshold=0.5):
        """
        모델 성능 지표(AUC, Precision, Recall) 계산
        """
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            "auc": float(roc_auc_score(y_true, y_proba)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0))
        }
        return metrics

    def add_performance_report(self, ref_metrics, cur_metrics):
        """
        성능 비교 및 알림 생성
        """
        self.report["performance"] = {
            "reference": ref_metrics,
            "current": cur_metrics,
            "drop": {
                k: ref_metrics[k] - cur_metrics[k] for k in ref_metrics
            }
        }
        
        # 성능 저하 알림 (AUC 기준)
        auc_drop = self.report["performance"]["drop"]["auc"]
        if auc_drop > self.threshold_performance_drop:
            self.report["alerts"].append({
                "type": "PERFORMANCE_DROP",
                "feature": "model_performance",
                "metric": "auc",
                "value": cur_metrics["auc"],
                "ref_value": ref_metrics["auc"],
                "drop": auc_drop,
                "threshold": self.threshold_performance_drop,
                "message": f"Significant performance drop detected (AUC drop: {auc_drop:.4f})"
            })

    def save_report(self, output_path="results/monitoring_report.json"):
        """결과를 JSON 파일로 저장"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=4, ensure_ascii=False)
        
        return path
