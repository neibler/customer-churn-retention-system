import pandas as pd
import numpy as np
from src.monitoring.drift_detector import DriftDetector
from pathlib import Path

def generate_dummy_monitoring_data():
    """대시보드 테스트를 위한 더미 모니터링 리포트 생성"""
    # 임의의 피처 데이터 생성
    np.random.seed(42)
    features = ["days_since_last_purchase", "total_sessions", "avg_session_duration", "purchase_frequency_change_4w", "cart_abandonment_rate"]
    
    # Reference (과거)
    ref_df = pd.DataFrame({
        col: np.random.normal(10, 2, 1000) for col in features
    })
    
    # Current (현재 - 일부 피처에 드리프트 발생시킴)
    curr_df = pd.DataFrame({
        col: np.random.normal(10 if col != "days_since_last_purchase" else 12, 2, 1000) for col in features
    })
    
    detector = DriftDetector()
    detector.run_monitoring(ref_df, curr_df, features)
    path = detector.save_report()
    print(f"Monitoring report saved to {path}")

if __name__ == "__main__":
    generate_dummy_monitoring_data()
