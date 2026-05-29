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
    
    def __init__(self, threshold_psi=0.2, threshold_ks=0.05):
        self.threshold_psi = threshold_psi
        self.threshold_ks = threshold_ks
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
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
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": []
        }

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

    def save_report(self, output_path="results/monitoring_report.json"):
        """결과를 JSON 파일로 저장"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=4, ensure_ascii=False)
        
        return path
