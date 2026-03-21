"""
utils/predictor.py
==================
Loads trained models and provides prediction helpers for the dashboard.
"""

import os, json, pickle
import numpy as np
import pandas as pd

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

_rf     = None
_iso    = None
_scaler = None
_report = None


def _load():
    global _rf, _iso, _scaler, _report
    if _rf is None:
        with open(os.path.join(MODEL_DIR, "rf_model.pkl"),     "rb") as f: _rf     = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "iso_model.pkl"),    "rb") as f: _iso    = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"),       "rb") as f: _scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "model_report.json"),"r") as f: _report = json.load(f)


def get_report():
    _load()
    return _report


def load_data():
    sensor = pd.read_csv(os.path.join(DATA_DIR, "sensor_data.csv"),   parse_dates=["timestamp"])
    meta   = pd.read_csv(os.path.join(DATA_DIR, "machine_metadata.csv"))
    return sensor, meta


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    BASE_FEATS = ["temperature", "vibration", "rpm", "pressure"]
    df = df.sort_values("timestamp").copy()
    for col in BASE_FEATS:
        df[f"{col}_roll6_mean"]  = df[col].rolling(6,  min_periods=1).mean()
        df[f"{col}_roll24_mean"] = df[col].rolling(24, min_periods=1).mean()
        df[f"{col}_roll6_std"]   = df[col].rolling(6,  min_periods=1).std().fillna(0)
    df["vib_x_temp"] = df["vibration"] * df["temperature"]
    return df


def predict_machine(machine_df: pd.DataFrame):
    """
    Given time-series rows for ONE machine, return:
      - risk_score   : float 0-1  (RF probability of fault)
      - iso_score    : float      (negative = more anomalous)
      - risk_level   : str        ('CRITICAL' / 'WARNING' / 'NORMAL')
      - recent_proba : array of fault probabilities over last 168 rows (1 week)
    """
    _load()
    feat_cols = _report["feature_columns"]

    df_feat = engineer_features(machine_df)
    # keep only rows that have all feature columns
    available = [c for c in feat_cols if c in df_feat.columns]
    X_raw = df_feat[available].fillna(0).values

    if len(X_raw) == 0:
        return 0.0, 0.0, "NORMAL", []

    X_scaled = _scaler.transform(X_raw)

    rf_proba  = _rf.predict_proba(X_scaled)[:, 1]
    iso_score = _iso.decision_function(X_scaled).mean()

    # risk_score = average of last 24 readings
    recent_proba = rf_proba[-168:] if len(rf_proba) >= 168 else rf_proba
    risk_score   = float(rf_proba[-24:].mean())

    if risk_score >= 0.35:
        level = "CRITICAL"
    elif risk_score >= 0.15:
        level = "WARNING"
    else:
        level = "NORMAL"

    return risk_score, float(iso_score), level, recent_proba.tolist()


def get_all_machine_risk(sensor_df, meta_df):
    """Returns a summary DataFrame with risk for every machine."""
    rows = []
    for mid, grp in sensor_df.groupby("machine_id"):
        risk, iso, level, _ = predict_machine(grp)
        meta_row = meta_df[meta_df["machine_id"] == mid].iloc[0]
        rows.append({
            "machine_id":   mid,
            "machine_type": meta_row["machine_type"],
            "location":     meta_row["location"],
            "age_years":    meta_row["age_years"],
            "risk_score":   round(risk * 100, 1),
            "iso_score":    round(iso, 4),
            "risk_level":   level,
        })
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)
