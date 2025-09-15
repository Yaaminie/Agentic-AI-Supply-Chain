from __future__ import annotations
import pandas as pd
from typing import List, Optional
import numpy as np

def add_time_features_if_present(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    time_col: Optional[str] = None
    for c in df2.columns:
        lc = c.lower()
        if lc in {"timestamp", "time", "date", "datetime"} or ("time" in lc or "date" in lc):
            time_col = c
            break
    if time_col:
        dt = pd.to_datetime(df2[time_col], errors="coerce")
        df2["hour"] = dt.dt.hour
        df2["dayofweek"] = dt.dt.dayofweek
        df2["month"] = dt.dt.month
        df2["is_weekend"] = (df2["dayofweek"] >= 5).astype("float64")
    return df2

def infer_numeric_features(df: pd.DataFrame, target_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != target_col]
