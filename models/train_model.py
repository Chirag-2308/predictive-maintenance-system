"""
train_model.py
==============
Trains two complementary models:
  1. Random Forest Classifier  – supervised fault prediction (uses fault_label)
  2. Isolation Forest          – unsupervised anomaly detector (no labels needed)

Saves:
  models/rf_model.pkl
  models/iso_model.pkl
  models/scaler.pkl
  models/model_report.json
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SENSOR_CSV   = os.path.join(DATA_DIR, "sensor_data.csv")
META_CSV     = os.path.join(DATA_DIR, "machine_metadata.csv")
RF_PATH      = os.path.join(MODEL_DIR, "rf_model.pkl")
ISO_PATH     = os.path.join(MODEL_DIR, "iso_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
REPORT_PATH  = os.path.join(MODEL_DIR, "model_report.json")

FEATURES = ["temperature", "vibration", "rpm", "pressure"]

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics as extra features per machine."""
    df = df.sort_values(["machine_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    result_frames = []
    for mid, grp in df.groupby("machine_id"):
        grp = grp.copy().sort_values("timestamp")
        for col in FEATURES:
            grp[f"{col}_roll6_mean"]  = grp[col].rolling(6,  min_periods=1).mean()
            grp[f"{col}_roll24_mean"] = grp[col].rolling(24, min_periods=1).mean()
            grp[f"{col}_roll6_std"]   = grp[col].rolling(6,  min_periods=1).std().fillna(0)
        # vibration-temperature interaction
        grp["vib_x_temp"] = grp["vibration"] * grp["temperature"]
        result_frames.append(grp)

    return pd.concat(result_frames, ignore_index=True)


def get_feature_cols(df: pd.DataFrame):
    base    = FEATURES
    rolling = [c for c in df.columns if "roll" in c]
    extra   = ["vib_x_temp"]
    return base + rolling + extra

# ─────────────────────────────────────────────
#  LOAD & PREPARE
# ─────────────────────────────────────────────
print("Loading data …")
sensor_df = pd.read_csv(SENSOR_CSV)
meta_df   = pd.read_csv(META_CSV)

print("Engineering features …")
sensor_df = engineer_features(sensor_df)
feat_cols  = get_feature_cols(sensor_df)

X = sensor_df[feat_cols].values
y = sensor_df["fault_label"].values

print(f"Dataset  : {len(X):,} samples  |  Features: {len(feat_cols)}")
print(f"Positives: {y.sum():,}  ({y.mean()*100:.2f}%)  |  Negatives: {(y==0).sum():,}")

# ─────────────────────────────────────────────
#  SCALE
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
#  1. RANDOM FOREST  (supervised)
# ─────────────────────────────────────────────
print("\nTraining Random Forest Classifier …")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",   # handles imbalance
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_proba)
except Exception:
    auc = 0.0

print(f"  Precision : {precision*100:.1f}%")
print(f"  Recall    : {recall*100:.1f}%")
print(f"  F1 Score  : {f1*100:.1f}%")
print(f"  ROC-AUC   : {auc:.4f}")

cm = confusion_matrix(y_test, y_pred).tolist()
print(f"  Confusion Matrix: {cm}")

# Feature importances
importances = dict(zip(feat_cols, rf.feature_importances_.tolist()))
top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

# ─────────────────────────────────────────────
#  2. ISOLATION FOREST  (unsupervised anomaly)
# ─────────────────────────────────────────────
print("\nTraining Isolation Forest (unsupervised) …")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.015,
    random_state=42,
    n_jobs=-1
)
# train only on normal data
X_normal = X_scaled[y == 0]
iso.fit(X_normal)

iso_pred_raw = iso.predict(X_test)
iso_pred     = np.where(iso_pred_raw == -1, 1, 0)   # -1 → anomaly → 1

iso_recall = recall_score(y_test, iso_pred, zero_division=0)
iso_prec   = precision_score(y_test, iso_pred, zero_division=0)
print(f"  Isolation Forest Recall    : {iso_recall*100:.1f}%")
print(f"  Isolation Forest Precision : {iso_prec*100:.1f}%")

# ─────────────────────────────────────────────
#  SAVE ARTEFACTS
# ─────────────────────────────────────────────
print("\nSaving models …")
with open(RF_PATH,     "wb") as f: pickle.dump(rf,     f)
with open(ISO_PATH,    "wb") as f: pickle.dump(iso,    f)
with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

report = {
    "random_forest": {
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1_score":   round(f1,        4),
        "roc_auc":    round(auc,       4),
        "confusion_matrix": cm,
        "top_features": top_features,
        "n_estimators": 200,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    },
    "isolation_forest": {
        "recall":     round(iso_recall, 4),
        "precision":  round(iso_prec,   4),
        "contamination": 0.015,
    },
    "feature_columns": feat_cols,
    "base_features":   FEATURES,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nAll artefacts saved to  {MODEL_DIR}/")
print("Training complete.")
