import pandas as pd
from supplychain_risk.model.trainer import train_model_and_explain

def test_trainer_smoke():
    df = pd.DataFrame({
        "f1":[1,2,3,4,5,6,7,8,9,10],
        "f2":[0.1,0.2,0.2,0.3,0.5,0.1,0.9,0.8,0.2,0.1],
        "risk_classification":["Low","Low","High","Low","Moderate","High","High","Moderate","Low","High"]
    })
    out = train_model_and_explain(df, target_col="risk_classification", random_state=0)
    assert "metrics" in out and "preds_df" in out
