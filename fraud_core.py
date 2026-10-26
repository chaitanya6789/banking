"""
fraud_core.py  -  all ML logic of the project (no UI code).

Mirrors Project_SRM_final.ipynb step by step:
  load -> drop Time -> 60/40 stratified split -> RobustScaler(Amount) ->
  3 feature partitions -> RF / XGB / DT sub-models -> weighted soft vote ->
  PSI drift audit -> strategies A (static), B (global), C (selective).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NOTEBOOK_ROWS = 186_283          # rows the Colab notebook actually used
GRP1 = [f"V{i}" for i in range(1, 10)]                        # Partition 1
GRP2 = [f"V{i}" for i in range(10, 20)]                       # Partition 2
GRP3 = [f"V{i}" for i in range(20, 29)] + ["Scaled_Amount"]   # Partition 3
ALL_COLS = GRP1 + GRP2 + GRP3
PARTITIONS: Dict[str, List[str]] = {"P1": GRP1, "P2": GRP2, "P3": GRP3}
MODEL_NAMES = {"P1": "Random Forest", "P2": "XGBoost", "P3": "Decision Tree"}
RAW_REQUIRED = [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_CFG = dict(
    seed=42, rf_trees=60, rf_depth=6, xgb_trees=60, xgb_depth=4, dt_depth=6,
    global_trees=180, w1=0.30, w2=0.45, w3=0.25, threshold=0.50, psi_threshold=0.25,
)

LOCAL_CANDIDATES = [
    "creditcard.csv", "creditcard.zip", "creditcard.csv.zip",
    "data/creditcard.csv", "data/creditcard.zip", "data/creditcard.csv.zip",
]


# ----------------------------------------------------------------------------
# Data loading / preparation
# ----------------------------------------------------------------------------
def find_local_dataset():
    """Return the path of a dataset shipped inside the repo, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in LOCAL_CANDIDATES:
        for base in (here, os.getcwd()):
            full = os.path.join(base, p)
            if os.path.exists(full):
                return full
    return None


def load_dataset(source, nrows=None, filename: str = "") -> pd.DataFrame:
    """source = path or file-like object (csv or zip). nrows=None -> all rows."""
    name = (filename or (source if isinstance(source, str) else "")).lower()
    comp = "zip" if name.endswith(".zip") else "infer"
    df = pd.read_csv(source, nrows=nrows, compression=comp)
    missing = [c for c in RAW_REQUIRED + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing[:6]} ...")
    return df


def prepare_data(df_raw: pd.DataFrame, seed: int = 42) -> dict:
    """Notebook steps 2-3: clean, split 60/40, scale Amount (fit on reference only)."""
    df = df_raw.dropna().reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    df_proc = df.drop(columns=["Time"], errors="ignore")

    df_ref, df_prod = train_test_split(
        df_proc, test_size=0.40, stratify=df_proc["Class"], random_state=seed)
    df_ref, df_prod = df_ref.reset_index(drop=True), df_prod.reset_index(drop=True)

    scaler = RobustScaler().fit(df_ref[["Amount"]])
    for d in (df_ref, df_prod):
        d["Scaled_Amount"] = scaler.transform(d[["Amount"]])
    df_ref = df_ref.drop(columns=["Amount"])
    df_prod = df_prod.drop(columns=["Amount"])

    y_ref = df_ref["Class"].values
    pos_weight = float((y_ref == 0).sum() / max((y_ref == 1).sum(), 1))
    return dict(df_raw=df, df_ref=df_ref, df_prod=df_prod, scaler=scaler,
                pos_weight=pos_weight)


# ----------------------------------------------------------------------------
# Sub-models
# ----------------------------------------------------------------------------
def make_m1(cfg):
    return RandomForestClassifier(n_estimators=cfg["rf_trees"], max_depth=cfg["rf_depth"],
                                  class_weight="balanced", random_state=cfg["seed"], n_jobs=-1)


def make_m2(cfg, pos_weight):
    return XGBClassifier(n_estimators=cfg["xgb_trees"], max_depth=cfg["xgb_depth"],
                         scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=cfg["seed"], n_jobs=-1)


def make_m3(cfg):
    return DecisionTreeClassifier(max_depth=cfg["dt_depth"], class_weight="balanced",
                                  random_state=cfg["seed"])


def make_model(key, cfg, pos_weight):
    return {"P1": lambda: make_m1(cfg),
            "P2": lambda: make_m2(cfg, pos_weight),
            "P3": lambda: make_m3(cfg)}[key]()


def train_submodels(prep: dict, cfg: dict) -> Tuple[dict, dict]:
    X, y = prep["df_ref"].drop(columns=["Class"]), prep["df_ref"]["Class"].values
    models, times = {}, {}
    for key in ("P1", "P2", "P3"):
        t0 = time.time()
        m = make_model(key, cfg, prep["pos_weight"])
        m.fit(X[PARTITIONS[key]], y)
        models[key], times[key] = m, time.time() - t0
    return models, times


def normalise_weights(cfg) -> Tuple[float, float, float]:
    w = np.array([cfg["w1"], cfg["w2"], cfg["w3"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.array([0.30, 0.45, 0.25])
    return tuple(w.tolist())


def predict_ensemble(models: dict, data: pd.DataFrame, weights=(0.30, 0.45, 0.25),
                     threshold: float = 0.50):
    """Weighted soft voting. Returns (p1, p2, p3, final_prob, final_pred)."""
    p = [models[k].predict_proba(data[PARTITIONS[k]])[:, 1] for k in ("P1", "P2", "P3")]
    final = weights[0] * p[0] + weights[1] * p[1] + weights[2] * p[2]
    return p[0], p[1], p[2], final, (final >= threshold).astype(int)


def get_scores(y_true, y_pred) -> List[float]:
    return [precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)]


def baseline_table(models, prep, cfg) -> pd.DataFrame:
    X, y = prep["df_prod"].drop(columns=["Class"]), prep["df_prod"]["Class"].values
    p1, p2, p3, _, pred = predict_ensemble(models, X, normalise_weights(cfg), cfg["threshold"])
    rows = {
        "Sub-Model 1 (RF)": get_scores(y, (p1 >= 0.5).astype(int)),
        "Sub-Model 2 (XGB)": get_scores(y, (p2 >= 0.5).astype(int)),
        "Sub-Model 3 (DT)": get_scores(y, (p3 >= 0.5).astype(int)),
        "Weighted Ensemble": get_scores(y, pred),
    }
    return pd.DataFrame(rows, index=["Precision", "Recall", "F1-Score"]).T


# ----------------------------------------------------------------------------
# PSI drift engine
# ----------------------------------------------------------------------------
def calculate_psi(reference, production, num_bins: int = 10) -> float:
    r_counts, edges = np.histogram(reference, bins=num_bins)      # bins from OLD data
    p_counts, _ = np.histogram(production, bins=edges)            # same bins for NEW data
    r_pct = np.where(r_counts == 0, 1e-4, r_counts) / len(reference)
    p_pct = np.where(p_counts == 0, 1e-4, p_counts) / len(production)
    return float(np.sum((p_pct - r_pct) * np.log(p_pct / r_pct)))


def partition_psi(ref_df, prod_df, cols: Sequence[str], threshold=0.25):
    per_feature = {c: calculate_psi(ref_df[c], prod_df[c]) for c in cols}
    avg = float(np.mean(list(per_feature.values())))
    return avg >= threshold, avg, per_feature


def psi_state(psi: float, threshold: float = 0.25) -> str:
    return "DRIFT" if psi >= threshold else ("WARNING" if psi >= 0.10 else "STABLE")


def audit_all(ref_df, prod_df, threshold=0.25) -> dict:
    out = {}
    for k, cols in PARTITIONS.items():
        flag, avg, per = partition_psi(ref_df, prod_df, cols, threshold)
        out[k] = dict(flag=flag, psi=avg, per_feature=per)
    return out


def apply_drift(df_prod: pd.DataFrame, partitions: Sequence[str], scale=1.6,
                shift=2.0, noise=0.5, seed=42) -> pd.DataFrame:
    """x' = scale * x + N(shift, noise^2) on the chosen partitions (notebook cell 24)."""
    out = df_prod.copy()
    np.random.seed(seed)
    for k in partitions:
        for col in PARTITIONS[k]:
            out[col] = out[col] * scale + np.random.normal(shift, noise, size=len(out))
    return out


# ----------------------------------------------------------------------------
# Strategies A / B / C
# ----------------------------------------------------------------------------
def run_strategies(models, prep, drifted_prod, audit, cfg) -> dict:
    seed = cfg["seed"]
    w, thr = normalise_weights(cfg), cfg["threshold"]
    df_adapt, df_test = train_test_split(drifted_prod, test_size=0.50,
                                         stratify=drifted_prod["Class"], random_state=seed)
    X_adapt, y_adapt = df_adapt.drop(columns=["Class"]), df_adapt["Class"].values
    X_test, y_test = df_test.drop(columns=["Class"]), df_test["Class"].values

    res = {"split": dict(adapt_rows=len(df_adapt), adapt_frauds=int(y_adapt.sum()),
                         test_rows=len(df_test), test_frauds=int(y_test.sum()))}

    # A: static - do nothing
    _, _, _, prob_a, pred_a = predict_ensemble(models, X_test, w, thr)
    res["A"] = dict(scores=get_scores(y_test, pred_a), auc=roc_auc_score(y_test, prob_a),
                    time=0.0, cm=confusion_matrix(y_test, pred_a, labels=[0, 1]))

    # B: global retraining - one big XGBoost on all 29 features
    t0 = time.time()
    gm = XGBClassifier(n_estimators=cfg["global_trees"], max_depth=cfg["xgb_depth"],
                       scale_pos_weight=prep["pos_weight"], eval_metric="logloss",
                       random_state=seed, n_jobs=-1)
    gm.fit(X_adapt[ALL_COLS], y_adapt)
    t_b = time.time() - t0
    prob_b = gm.predict_proba(X_test[ALL_COLS])[:, 1]
    pred_b = (prob_b >= thr).astype(int)
    res["B"] = dict(scores=get_scores(y_test, pred_b), auc=roc_auc_score(y_test, prob_b),
                    time=t_b, cm=confusion_matrix(y_test, pred_b, labels=[0, 1]))

    # C: selective retraining - only the sub-models of flagged partitions
    flagged = [k for k in PARTITIONS if audit[k]["flag"]]
    new_models = dict(models)
    t0 = time.time()
    for k in flagged:
        m = make_model(k, cfg, prep["pos_weight"])
        m.fit(X_adapt[PARTITIONS[k]], y_adapt)
        new_models[k] = m
    t_c = time.time() - t0
    _, _, _, prob_c, pred_c = predict_ensemble(new_models, X_test, w, thr)
    res["C"] = dict(scores=get_scores(y_test, pred_c), auc=roc_auc_score(y_test, prob_c),
                    time=t_c, cm=confusion_matrix(y_test, pred_c, labels=[0, 1]),
                    flagged=flagged)
    res["saved_pct"] = (1 - t_c / t_b) * 100 if t_b > 0 else 0.0
    res["updated_models"] = new_models
    return res


def results_frame(res: dict) -> pd.DataFrame:
    names = {"A": "A: Static", "B": "B: Global retraining", "C": "C: Selective retraining"}
    rows = []
    for k in "ABC":
        p, r, f = res[k]["scores"]
        rows.append([p, r, f, res[k]["auc"], res[k]["time"]])
    return pd.DataFrame(rows, index=[names[k] for k in "ABC"],
                        columns=["Precision", "Recall", "F1-Score", "ROC-AUC",
                                 "Retrain time (s)"])
