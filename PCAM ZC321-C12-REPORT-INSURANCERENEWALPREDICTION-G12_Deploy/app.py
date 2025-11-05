import streamlit as st
import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from scipy.special import expit

st.set_page_config(page_title="Insurance Renewal Predictor", layout="wide")

# ----------------------------
# Bundle & helpers
# ----------------------------
BUNDLE_DIR = Path("model_bundle_serial_smote_v1")

@st.cache_resource
def load_joblib_safe(p: Path):
    return joblib.load(str(p))

def load_bundle(bundle_dir: Path):
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {bundle_dir}")
    meta = json.loads(meta_path.read_text())

    scaler = None
    scaler_name = meta.get("scaler_file", "scaler.joblib")
    scaler_path = bundle_dir / scaler_name
    if scaler_path.exists():
        scaler = load_joblib_safe(scaler_path)

    stages_meta = []
    if "stages" in meta and isinstance(meta["stages"], list) and meta["stages"]:
        for s in meta["stages"]:
            fname = s.get("model_file") or (s.get("name") + ".joblib")
            p = bundle_dir / fname
            if not p.exists():
                raise FileNotFoundError(f"Stage file missing: {p}")
            thr = meta.get("stages_thresholds", {}).get(p.name, None)
            stages_meta.append({"path": str(p), "filename": p.name, "threshold": (float(thr) if thr is not None else None)})
    else:
        found = sorted(bundle_dir.glob("stage_*.joblib"))
        for p in found:
            thr = meta.get("stages_thresholds", {}).get(p.name, None)
            stages_meta.append({"path": str(p), "filename": p.name, "threshold": (float(thr) if thr is not None else None)})

    return {"meta": meta, "scaler": scaler, "stages": stages_meta, "bundle_dir": str(bundle_dir)}

def unwrap_model(obj):
    if isinstance(obj, dict):
        if "model" in obj:
            m = obj["model"]
            if isinstance(m, dict) and "model" in m:
                return m["model"]
            return m
        if "artifacts" in obj and isinstance(obj["artifacts"], dict):
            for v in obj["artifacts"].values():
                if hasattr(v, "predict_proba") or hasattr(v, "decision_function") or hasattr(v, "predict"):
                    return v
        for v in obj.values():
            if hasattr(v, "predict_proba") or hasattr(v, "decision_function") or hasattr(v, "predict"):
                return v
        return obj
    return obj

def compute_model_probs(model_obj, X):
    if model_obj is None:
        return np.zeros(len(X), dtype=float)
    model = unwrap_model(model_obj)
    if isinstance(model, (str, Path)):
        model = load_joblib_safe(Path(model))

    X_input = X
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_input)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1].astype(float)
            else:
                return p.reshape(-1).astype(float)
        if hasattr(model, "decision_function"):
            df_out = np.asarray(model.decision_function(X_input)).reshape(-1)
            mn, mx = np.nanmin(df_out), np.nanmax(df_out)
            if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                return ((df_out - mn) / (mx - mn)).astype(float)
            else:
                return expit(df_out).astype(float)
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X_input)).reshape(-1).astype(float)
    except Exception:
        try:
            arr = X_input.values if hasattr(X_input, "values") else np.asarray(X_input)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(arr); p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] >= 2: return p[:,1].astype(float)
                return p.reshape(-1).astype(float)
            if hasattr(model, "decision_function"):
                df_out = np.asarray(model.decision_function(arr)).reshape(-1)
                mn, mx = np.nanmin(df_out), np.nanmax(df_out)
                if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                    return ((df_out - mn) / (mx - mn)).astype(float)
                else:
                    return expit(df_out).astype(float)
            if hasattr(model, "predict"):
                return np.asarray(model.predict(arr)).reshape(-1).astype(float)
        except Exception:
            return np.zeros(len(X), dtype=float)
    return np.zeros(len(X), dtype=float)

# ----------------------------
# Load bundle
# ----------------------------
try:
    bundle = load_bundle(BUNDLE_DIR)
    meta = bundle["meta"]
    scaler = bundle["scaler"]
    stages_meta = bundle["stages"]
except Exception as e:
    st.error(f"Failed to load model bundle: {e}")
    st.stop()

feature_names = meta.get("feature_names", [])
ui_feature_names =  feature_names
feature_names = [f for f in feature_names if f.lower() not in ("id", "age_in_days")]
if "age_in_years" not in feature_names:
    feature_names.append("age_in_years")

label_map = meta.get("label_map", {"model_label_1": "non_renew", "model_label_0": "renew"})
label_for_1 = label_map.get("model_label_1", "non_renew")
label_for_0 = label_map.get("model_label_0", "renew")

loaded_stages = []
for s in stages_meta:
    p = Path(s["path"])
    obj = load_joblib_safe(p)
    loaded_stages.append({"filename": p.name, "model_obj": obj, "threshold": s.get("threshold")})

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔍 PredictMyPolicy")
st.write("Upload a CSV containing the features. `id` is optional and preserved for output. `age_in_days` is automatically converted to `age_in_years` if present.")

include_stage_probs = st.checkbox("Include per-stage probabilities in output (advanced)", value=False)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file. Use the expected feature list below to prepare your file.")
    st.subheader("Expected model features:")
    st.write(ui_feature_names)
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write(f"Uploaded file: {df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head(20), hide_index=True)

id_col_candidates = [c for c in df.columns if "id" in c.lower() or "policy" in c.lower()]
id_col = id_col_candidates[0] if id_col_candidates else None
if id_col:
    id_series = df[id_col].reset_index(drop=True)
    st.info(f"Detected ID column `{id_col}` — preserved for output.")
else:
    id_series = pd.Series(range(1, len(df) + 1), name="row_id")

if "age_in_days" in df.columns and "age_in_years" not in df.columns:
    df["age_in_years"] = df["age_in_days"] / 365.0
    st.info("Converted `age_in_days` → `age_in_years`.")

missing = [f for f in feature_names if f not in df.columns]
if missing:
    st.error(f"Missing required features: {missing}")
    st.stop()

X = df[feature_names].copy()
if X.isnull().any().any():
    X = X.fillna(X.median())

if scaler is not None:
    try:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names, index=X.index)
    except ValueError as e:
        st.error(f"Scaler shape mismatch: {e}")
        st.stop()
else:
    X_scaled = X

prob_matrix = []
stage_names = []
for s in loaded_stages:
    stage_names.append(s["filename"])
    probs = compute_model_probs(s["model_obj"], X_scaled)
    if len(probs) != len(X_scaled):
        probs = np.resize(probs, len(X_scaled))
    prob_matrix.append(probs)

prob_matrix = np.vstack(prob_matrix).T
thresholds = [meta.get("stages_thresholds", {}).get(s["filename"], s.get("threshold", 0.5)) for s in loaded_stages]
votes = (prob_matrix >= np.array(thresholds)[None, :]).astype(int)

model_bit = (votes.sum(axis=1) >= 1).astype(int)
pseudo_prob = votes.sum(axis=1) / max(1, votes.shape[1])
pred_labels = np.where(model_bit == 1, label_for_1, label_for_0)

# ----------------------------
# Output Data
# ----------------------------
out = pd.DataFrame({
    id_series.name: id_series.values,
    "prediction": pred_labels,
    "renew_score": np.round(1.0 - pseudo_prob, 6),
    "non_renew_score": np.round(pseudo_prob, 6)
})

if include_stage_probs:
    for i, name in enumerate(stage_names):
        out[f"prob_{name}"] = np.round(prob_matrix[:, i], 6)

st.success("✅ Predictions ready!")
st.dataframe(out.head(20), use_container_width=True, hide_index=True)

# Download button
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download predictions", data=csv_bytes, file_name="renewal_predictions.csv", mime="text/csv", key="download_preds_v2")

st.markdown("---")
st.markdown("Notes: Bundle must include `meta.json`, `scaler.joblib` (optional), and `stage_*.joblib` models.")
