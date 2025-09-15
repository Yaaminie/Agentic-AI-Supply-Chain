from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update
from flask import request, jsonify

from .config import settings
from .logging_config import setup_logging
from .utils.io import load_csv
from .model.trainer import train_model_and_explain
from .model.shap_utils import is_multiclass_shap
from .notify.policy import policy_recommendation
from .notify.llm import llm_manager_summary, llm_customer_message
from .notify.slack import send_slack

setup_logging()
log = logging.getLogger(__name__)

# ---------- small plot helpers ----------
def confusion_matrix_fig(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    fig = px.imshow(cm, x=class_names, y=class_names, text_auto=True, color_continuous_scale="Blues",
                    title="Confusion Matrix")
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
    return fig

def feature_importance_fig(feature_names: List[str], importances: np.ndarray, top_n: int = 20) -> go.Figure:
    idx = np.argsort(importances)[::-1][:top_n]
    data = pd.DataFrame({"feature": np.array(feature_names)[idx], "importance": importances[idx]}) \
             .sort_values("importance", ascending=True)
    return px.bar(data, x="importance", y="feature", orientation="h", title="XGBoost Feature Importance (gain)")

def learning_curve_fig(evals_result: dict, metric_key: str) -> go.Figure:
    if not evals_result or "validation_0" not in evals_result or metric_key not in evals_result["validation_0"]:
        return go.Figure()
    train_metric = evals_result["validation_0"][metric_key]
    valid_metric = evals_result.get("validation_1", {}).get(metric_key, None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_metric, name="train"))
    if valid_metric:
        fig.add_trace(go.Scatter(y=valid_metric, name="valid"))
    fig.update_layout(title=f"Learning Curve ({metric_key})", xaxis_title="Boosting rounds", yaxis_title=metric_key)
    return fig

def shap_global_bar_fig(shap_values_norm, X_sample: pd.DataFrame) -> go.Figure:
    if isinstance(shap_values_norm, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_norm], axis=0)
    else:
        mean_abs = np.abs(shap_values_norm).mean(axis=0)
    data = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs}) \
             .sort_values("mean_abs_shap", ascending=True).tail(25)
    return px.bar(data, x="mean_abs_shap", y="feature", orientation="h", title="SHAP mean |impact| (global)")

def shap_dependence_fig(shap_values_norm, X_sample: pd.DataFrame, feature_name: str, class_index: Optional[int]) -> go.Figure:
    if isinstance(shap_values_norm, list):
        ci = 0 if class_index is None else int(class_index)
        ci = max(0, min(ci, len(shap_values_norm) - 1))
        sv = shap_values_norm[ci]
        suffix = f" (class {ci})"
    else:
        sv = shap_values_norm
        suffix = ""
    if feature_name not in X_sample.columns:
        return go.Figure()
    i = int(list(X_sample.columns).index(feature_name))
    return px.scatter(x=X_sample[feature_name], y=sv[:, i],
                      labels={"x": feature_name, "y": "SHAP value"},
                      title=f"SHAP Dependence: {feature_name}{suffix}")

# ---------- Dash App ----------
app = Dash(__name__)
app.title = "Supply Chain Risk — EDA & XGBoost"

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto", "margin": "16px"},
    children=[
        html.H2("Supply Chain Risk Dashboard (XGBoost + SHAP)"),
        html.Div([
            dcc.Input(id="csv-path", type="text", placeholder="Path to CSV",
                      value=settings.csv_path, style={"width": "60%"}),
            html.Button("Load Data", id="btn-load", n_clicks=0, style={"marginLeft": "8px"}),
            html.Button("Train Model", id="btn-train", n_clicks=0, style={"marginLeft": "8px"}),
            html.Button("Notify High-Risk (Agentic)", id="btn-notify", n_clicks=0, style={"marginLeft": "8px"}),
            html.Span(id="status-msg", style={"marginLeft": "12px", "color": "#666"})
        ], style={"marginBottom": "12px"}),

        html.Hr(),

        dcc.Tabs(id="tabs", value="tab-eda", children=[
            dcc.Tab(label="EDA", value="tab-eda", children=[
                html.Div(id="eda-summary", style={"margin": "12px 0"}),
                dcc.Graph(id="missingness"),
                dcc.Graph(id="corr-heatmap"),
                html.Div(id="eda-histograms"),
                dcc.Graph(id="latlon-scatter")
            ]),
            dcc.Tab(label="Model", value="tab-model", children=[
                html.Div(id="model-metrics", style={"whiteSpace": "pre", "fontFamily": "monospace"}),
                dcc.Graph(id="cm-fig"),
                dcc.Graph(id="fi-fig"),
                dcc.Graph(id="lc-fig"),
                html.H4("Sample Predictions"),
                dash_table.DataTable(
                    id="preds-table",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "minWidth": "100px", "whiteSpace": "normal"},
                )
            ]),
            dcc.Tab(label="Explainability", value="tab-xai", children=[
                html.Div(id="xai-top", style={"margin": "12px 0"}),
                dcc.Graph(id="shap-global"),
                html.Div([
                    html.Label("Dependence feature:"),
                    dcc.Dropdown(id="dep-feature", options=[], value=None, style={"width": "300px"}),
                    html.Label("Class (for multiclass):", style={"marginLeft": "16px"}),
                    dcc.Dropdown(id="dep-class", options=[], value=None, style={"width": "200px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "margin": "8px 0"}),
                dcc.Graph(id="shap-dependence"),
            ]),
        ]),

        dcc.Store(id="store-df"),
        dcc.Store(id="store-model-out"),
    ]
)

# ---------- Stores for SHAP heavy objects ----------
X_SAMPLE_CACHE: Optional[pd.DataFrame] = None
SHAP_VALUES_CACHE = None

# ---------- Callbacks ----------
@app.callback(
    Output("store-df", "data"),
    Output("status-msg", "children", allow_duplicate=True),
    Input("btn-load", "n_clicks"),
    State("csv-path", "value"),
    prevent_initial_call=True
)
def load_data_cb(n_clicks: int, path: str):
    try:
        df = load_csv(path)
        if settings.target_col not in df.columns:
            return no_update, f"❌ '{settings.target_col}' not found."
        if df[settings.target_col].dtype != "object" and not str(df[settings.target_col].dtype).startswith("category"):
            return no_update, f"❌ '{settings.target_col}' must be categorical."
        return df.to_dict("records"), f"✅ Loaded {len(df)} rows."
    except Exception as e:
        log.exception("Load error")
        return no_update, f"❌ Error loading CSV: {e}"

@app.callback(
    Output("eda-summary", "children"),
    Output("missingness", "figure"),
    Output("corr-heatmap", "figure"),
    Output("eda-histograms", "children"),
    Output("latlon-scatter", "figure"),
    Input("store-df", "data"),
    prevent_initial_call=True
)
def update_eda(df_records):
    df = pd.DataFrame(df_records)
    rows, cols = df.shape
    num = df.select_dtypes(include=[np.number]).shape[1]
    obj = df.select_dtypes(include=["object", "category"]).shape[1]
    miss = df.isna().mean().mean() * 100
    summary = f"{rows} rows × {cols} columns  |  numeric: {num}, categorical: {obj}  |  overall missing: {miss:.1f}%"

    miss_df = df.isna().mean().sort_values(ascending=False) * 100
    miss_fig = px.bar(miss_df.reset_index().rename(columns={"index":"column", 0:"missing_%"}),
                      x="column", y="missing_%", title="Missing values (%)")
    miss_fig.update_layout(xaxis_tickangle=45)

    corr = df.corr(numeric_only=True)
    corr = corr.iloc[:25, :25] if corr.shape[0] > 25 else corr
    corr_fig = px.imshow(corr, aspect="auto", title="Correlation heatmap (numeric)") if corr.size else go.Figure()

    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_plot = [c for c in numeric_df.columns if c != settings.target_col and df[c].nunique(dropna=True) > 5][:6]
    hist_graphs = [dcc.Graph(figure=px.histogram(df, x=c, nbins=30, title=f"{c} (distribution)")) for c in cols_to_plot]

    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower()]
    latlon_fig = go.Figure()
    if lat_candidates and lon_candidates and settings.target_col in df.columns:
        lat_col, lon_col = lat_candidates[0], lon_candidates[0]
        if df[lat_col].dtype.kind in "fc" and df[lon_col].dtype.kind in "fc":
            latlon_fig = px.scatter(df, x=lon_col, y=lat_col, color=settings.target_col,
                                    title=f"Lat/Lon colored by {settings.target_col}", opacity=0.7)
            latlon_fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude")

    return summary, miss_fig, corr_fig, hist_graphs, latlon_fig

@app.callback(
    Output("store-model-out", "data"),
    Output("status-msg", "children", allow_duplicate=True),
    Input("btn-train", "n_clicks"),
    State("store-df", "data"),
    prevent_initial_call=True
)
def train_model_cb(n_clicks: int, df_records):
    if df_records is None:
        return no_update, "❌ Load data first."
    try:
        df = pd.DataFrame(df_records)
        out = train_model_and_explain(df, settings.target_col, settings.random_state)
        global X_SAMPLE_CACHE, SHAP_VALUES_CACHE
        X_SAMPLE_CACHE = out["X_sample"]
        SHAP_VALUES_CACHE = out["shap_values"]

        serializable = {
            "feature_cols": out["feature_cols"],
            "class_names": out["class_names"],
            "metrics": out["metrics"],
            "report_txt": out["report_txt"],
            "cm": out["cm"].tolist(),
            "feature_importances": out["feature_importances"].tolist(),
            "evals_result": out["evals_result"],
            "preds_df": out["preds_df"].to_dict("records"),
        }
        return serializable, "✅ Model trained."
    except Exception as e:
        log.exception("Training error")
        return no_update, f"❌ Training error: {e}"

@app.callback(
    Output("model-metrics", "children"),
    Output("cm-fig", "figure"),
    Output("fi-fig", "figure"),
    Output("lc-fig", "figure"),
    Output("preds-table", "columns"),
    Output("preds-table", "data"),
    Input("store-model-out", "data"),
    prevent_initial_call=True
)
def update_model_outputs(model_out):
    if model_out is None:
        return "", go.Figure(), go.Figure(), go.Figure(), [], []
    m = model_out["metrics"]
    report_txt = model_out["report_txt"]
    metrics_str = (
        f"Accuracy: {m['accuracy']:.4f}\n"
        f"F1 (macro): {m['f1_macro']:.4f}\n"
        + (f"AUC: {m['auc']:.4f}\n" if m["auc"] is not None else "AUC: n/a\n")
        + "\n" + report_txt
    )
    cm_fig = confusion_matrix_fig(np.array(model_out["cm"]), model_out["class_names"])
    fi_fig = feature_importance_fig(model_out["feature_cols"], np.array(model_out["feature_importances"]))
    metric_key = "logloss" if len(model_out["class_names"]) == 2 else "mlogloss"
    lc_fig = learning_curve_fig(model_out["evals_result"], metric_key)

    preds_df = pd.DataFrame(model_out["preds_df"])
    columns = [{"name": c, "id": c, "type": "numeric" if preds_df[c].dtype.kind in "fc" else "text"} for c in preds_df.columns]
    data = preds_df.to_dict("records")
    return metrics_str, cm_fig, fi_fig, lc_fig, columns, data

@app.callback(
    Output("xai-top", "children"),
    Output("shap-global", "figure"),
    Output("dep-feature", "options"),
    Output("dep-feature", "value"),
    Output("dep-class", "options"),
    Output("dep-class", "value"),
    Input("store-model-out", "data"),
    prevent_initial_call=True
)
def update_xai_top(model_out):
    if model_out is None or X_SAMPLE_CACHE is None or SHAP_VALUES_CACHE is None:
        return "Train the model to view SHAP.", go.Figure(), [], None, [], None
    X_sample = X_SAMPLE_CACHE
    shap_norm = SHAP_VALUES_CACHE
    class_names = model_out["class_names"]

    try:
        global_fig = shap_global_bar_fig(shap_norm, X_sample)
    except Exception as e:
        return f"SHAP error: {e}", go.Figure(), [], None, [], None

    dep_feature_opts = [{"label": c, "value": c} for c in X_sample.columns]
    dep_feature_val = X_sample.columns[0] if len(X_sample.columns) else None

    if is_multiclass_shap(shap_norm):
        dep_class_opts = [{"label": n, "value": i} for i, n in enumerate(class_names)]
        dep_class_val = dep_class_opts[0]["value"] if dep_class_opts else None
    else:
        dep_class_opts, dep_class_val = [], None

    return (
        f"Global explainability across {len(X_sample)} sampled test rows.",
        global_fig, dep_feature_opts, dep_feature_val, dep_class_opts, dep_class_val
    )

@app.callback(
    Output("status-msg", "children", allow_duplicate=True),
    Input("btn-notify", "n_clicks"),
    State("store-model-out", "data"),
    prevent_initial_call=True
)
def notify_high_risk(n_clicks: int, model_out):
    if not model_out:
        return "❌ Train model first."

    preds_df = pd.DataFrame(model_out["preds_df"])
    class_names = model_out["class_names"]
    high_label = next((c for c in class_names if c.lower().startswith("high")), None)
    if not high_label:
        return "ℹ️ No 'High' class found in labels."

    # use predicted label + prob threshold union
    high_rows = preds_df[preds_df["y_pred"] == high_label].copy()
    prob_col = next((f"proba_{c}" for c in class_names if c.lower().startswith("high")), None)
    if prob_col and prob_col in preds_df.columns:
        high_rows = pd.concat([high_rows, preds_df[preds_df[prob_col] >= 0.70]]).drop_duplicates()

    if high_rows.empty:
        return "✅ No high-risk predictions in current sample."

    high_rows = high_rows.head(3).reset_index(drop=True)
    sent = 0
    for idx, row in high_rows.iterrows():
        context = {
            "risk_label": "High",
            "p_high": float(row.get(prob_col, 0.83) if prob_col else 0.83),
            "order_id": f"SO-DEMO-{idx+1}",
            "customer_tier": "GOLD",
            "promised_by": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
            "item_priority": "High",
            "promised_days_left": 0,
            "dc_stock": 0,
            "top_features": [{"name": n, "value": "—", "direction": "risk_up"}
                             for n in model_out["feature_cols"][:3]],
        }
        reco = policy_recommendation(context)
        context["policy_recommendation"] = reco
        mgr_msg = llm_manager_summary(context)
        cust_msg = llm_customer_message(context)
        send_slack(f":rotating_light: *Supply Risk Alert*\n{mgr_msg}")
        send_slack(f":package: *Customer Update*\n{cust_msg}")
        sent += 2

    return f"✅ Sent {sent} Slack messages for {len(high_rows)} high-risk cases."

# ----------- Tiny REST endpoint for MuleSoft / external calls -----------
@app.server.route("/agent/summarize", methods=["POST"])
def agent_summarize():
    payload = request.get_json(force=True) or {}
    reco = policy_recommendation(payload)
    payload["policy_recommendation"] = reco
    mgr = llm_manager_summary(payload)
    cust = llm_customer_message(payload)
    return jsonify({
        "recommended_action": reco,
        "manager_message": mgr,
        "customer_message": cust
    })

def run():
    app.run(host=settings.host, port=settings.port, debug=settings.debug)

if __name__ == "__main__":
    run()