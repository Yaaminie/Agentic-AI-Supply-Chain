from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap

from ..config import settings
from ..utils.features import add_time_features_if_present, infer_numeric_features
from .shap_utils import shap_to_arrays

def stratified_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_state)
    for tr, te in sss.split(X, y):
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    raise RuntimeError("Stratified split failed.")

def classwise_sample_weights(y_int: pd.Series) -> pd.Series:
    counts = y_int.value_counts()
    inv = 1.0 / counts
    w = y_int.map(inv).astype(float)
    w *= (len(w) / float(w.sum()))
    return w

def train_model_and_explain(df: pd.DataFrame, target_col: str = None, random_state: int = None) -> dict:
    target_col = target_col or settings.target_col
    random_state = random_state or settings.random_state

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if df[target_col].dtype != "object" and not str(df[target_col].dtype).startswith("category"):
        raise ValueError(f"Target '{target_col}' must be categorical (e.g., 'Low', 'Moderate', 'High').")

    df2 = add_time_features_if_present(df)
    feature_cols = infer_numeric_features(df2, target_col)
    if not feature_cols:
        raise ValueError("No numeric features found.")

    X = df2[feature_cols].copy()
    le = LabelEncoder()
    y_arr = np.asarray(le.fit_transform(df2[target_col]), dtype=int)
    y_int = pd.Series(y_arr, index=df2.index)
    class_names: List[str] = list(le.classes_)

    X_train, X_test, y_train, y_test = stratified_split(X, y_int, 0.8, random_state)
    imp = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(np.asarray(imp.fit_transform(X_train)), columns=feature_cols, index=X_train.index)
    X_test_i  = pd.DataFrame(np.asarray(imp.transform(X_test)),  columns=feature_cols, index=X_test.index)

    weights = classwise_sample_weights(y_train)
    n_classes = int(np.unique(y_train).shape[0])
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"

    model = XGBClassifier(
        objective=objective, n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist",
        random_state=random_state, eval_metric=eval_metric, n_jobs=-1
    )
    model.fit(
        X_train_i, y_train.to_numpy(),
        sample_weight=weights.to_numpy(),
        eval_set=[(X_train_i, y_train.to_numpy()), (X_test_i, y_test.to_numpy())],
        verbose=False
    )
    evals_result = model.evals_result()

    y_pred = model.predict(X_test_i)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    try:
        if n_classes == 2:
            y_proba = model.predict_proba(X_test_i)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
        else:
            y_proba = model.predict_proba(X_test_i)
            auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
    except Exception:
        auc = None

    report_txt = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fi = model.feature_importances_

    # SHAP — robust across versions
    try:
        bg = X_train_i.sample(min(500, len(X_train_i)), random_state=random_state)
        explainer = shap.TreeExplainer(model, data=bg, feature_names=feature_cols)
        X_sample = X_test_i.sample(min(1000, len(X_test_i)), random_state=random_state)
        shap_raw = explainer.shap_values(X_sample)
    except Exception:
        X_sample = X_test_i.sample(min(1000, len(X_test_i)), random_state=random_state)
        explainer = shap.Explainer(model, X_train_i)
        shap_raw = explainer(X_sample)

    shap_norm = shap_to_arrays(shap_raw)

    proba_sample = model.predict_proba(X_sample)
    pred_int = (np.argmax(proba_sample, axis=1) if proba_sample.ndim == 2
                else (proba_sample > 0.5).astype(int))
    pred_labels = [class_names[i] for i in pred_int]
    true_labels = [class_names[i] for i in y_test.loc[X_sample.index]]

    preds_df = pd.DataFrame({"y_true": true_labels, "y_pred": pred_labels}, index=X_sample.index)
    if proba_sample.ndim == 2:
        for j, cname in enumerate(class_names):
            preds_df[f"proba_{cname}"] = proba_sample[:, j].astype(float)
    else:
        preds_df["proba_positive"] = proba_sample.astype(float)

    return {
        "feature_cols": feature_cols,
        "class_names": class_names,
        "X_sample": X_sample,
        "shap_values": shap_norm,
        "metrics": {"accuracy": acc, "f1_macro": f1m, "auc": auc},
        "report_txt": report_txt,
        "cm": cm,
        "feature_importances": fi,
        "evals_result": evals_result,
        "preds_df": preds_df.reset_index(drop=True)
    }
