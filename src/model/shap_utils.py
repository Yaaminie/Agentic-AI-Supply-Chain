from __future__ import annotations
import numpy as np

def shap_to_arrays(shap_values):
    """Normalize SHAP output shape: list[np.ndarray] (multiclass) or 2D np.ndarray."""
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values
    vals = np.asarray(vals) if not isinstance(vals, list) else vals
    if isinstance(vals, list):
        return [np.asarray(v) for v in vals]
    if isinstance(vals, np.ndarray) and vals.ndim == 3:
        n_classes = vals.shape[2]
        return [vals[:, :, k] for k in range(n_classes)]
    return vals

def is_multiclass_shap(shap_values_norm) -> bool:
    return isinstance(shap_values_norm, list)
