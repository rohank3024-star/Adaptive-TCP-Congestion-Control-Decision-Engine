#!/usr/bin/env python3
"""
Module M5: Machine Learning Pipeline
======================================
Trains Decision Tree (baseline) and Random Forest classifiers to predict
the best TCP congestion control variant from network condition features.

Evaluation includes:
  - Held-out network condition evaluation (not random split)
  - Classification metrics: accuracy, precision, recall, F1, confusion matrix
  - Feature importance analysis
  - Baseline comparison: always-Reno vs always-Cubic vs always-BBR vs ML

Author: Shreyans Bohra (24BCT0013) – Data & ML
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ML_CSV       = "data/dataset_b_ml.csv"
MODEL_DIR    = "models"
RESULTS_DIR  = "results"
FEATURE_COLS = ["bandwidth_mbps", "delay_ms", "loss_pct", "baseline_rtt"]
LABEL_COL    = "best_variant"
TCP_VARIANTS = ["bbr", "cubic", "reno"]  # Alphabetical for consistency
RANDOM_STATE = 42
TEST_SIZE    = 0.2  # 20% held-out conditions


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ml_data(filepath=ML_CSV):
    """Load Dataset B for ML training."""
    df = pd.read_csv(filepath)
    print(f"[LOAD] ML dataset: {df.shape[0]} conditions, {df.shape[1]} features")
    print(f"  Label distribution: {df[LABEL_COL].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Held-out split by network condition
# ---------------------------------------------------------------------------
def held_out_split(df, test_size=TEST_SIZE):
    """
    Split data by holding out entire network conditions.
    This prevents data leakage between similar experiments.
    
    Parameters
    ----------
    df        : pd.DataFrame – Dataset B
    test_size : float – fraction of conditions to hold out
    
    Returns
    -------
    X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    # Create condition group IDs
    df = df.copy()
    df["condition_id"] = (
        df["bandwidth_mbps"].astype(str) + "_" +
        df["delay_ms"].astype(str) + "_" +
        df["loss_pct"].astype(str)
    )
    
    # Group shuffle split – holds out entire conditions
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["condition_id"]))
    
    X_train = df.iloc[train_idx][FEATURE_COLS].values
    X_test  = df.iloc[test_idx][FEATURE_COLS].values
    y_train = df.iloc[train_idx][LABEL_COL].values
    y_test  = df.iloc[test_idx][LABEL_COL].values
    
    print(f"\n[SPLIT] Held-out condition split:")
    print(f"  Training:  {len(train_idx)} conditions ({len(train_idx)/len(df)*100:.0f}%)")
    print(f"  Test:      {len(test_idx)} conditions ({len(test_idx)/len(df)*100:.0f}%)")
    print(f"  Train labels: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  Test labels:  {pd.Series(y_test).value_counts().to_dict()}")
    
    # Verify no leakage
    train_conditions = set(df.iloc[train_idx]["condition_id"])
    test_conditions  = set(df.iloc[test_idx]["condition_id"])
    overlap = train_conditions & test_conditions
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} conditions appear in both sets."
    print(f"  ✓ No data leakage: 0 overlapping conditions")
    
    return X_train, X_test, y_train, y_test, train_idx, test_idx


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_decision_tree(X_train, y_train):
    """Train a baseline Decision Tree classifier."""
    dt = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)
    train_acc = dt.score(X_train, y_train)
    print(f"\n[DT] Decision Tree trained – Train accuracy: {train_acc:.4f}")
    return dt


def train_random_forest(X_train, y_train):
    """Train the primary Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    print(f"\n[RF] Random Forest trained – Train accuracy: {train_acc:.4f}")
    return rf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on held-out test conditions.
    
    Returns
    -------
    dict – evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred, labels=TCP_VARIANTS)
    report = classification_report(y_test, y_pred, labels=TCP_VARIANTS, zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"  {model_name} – Held-out Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels: {TCP_VARIANTS}")
    for row in cm:
        print(f"    {row}")
    print(f"\n  Classification Report:")
    print(report)
    
    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "actual": y_test.tolist()
    }


def get_feature_importance(model, feature_names=FEATURE_COLS):
    """Extract and display feature importance from a tree-based model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n  Feature Importance:")
    result = {}
    for i in indices:
        print(f"    {feature_names[i]:20s}: {importances[i]:.4f}")
        result[feature_names[i]] = round(float(importances[i]), 4)
    
    return result


# ---------------------------------------------------------------------------
# Baseline comparison (the key evaluation)
# ---------------------------------------------------------------------------
def baseline_comparison(df, test_idx):
    """
    Compare average throughput on held-out conditions for:
      1. Always Reno
      2. Always Cubic
      3. Always BBR
      4. ML Decision Engine (adaptive selection)
    
    This is the central claim evaluation of the project.
    
    Parameters
    ----------
    df       : pd.DataFrame – Dataset B with predictions added
    test_idx : array – indices of held-out test conditions
    
    Returns
    -------
    dict – throughput comparison results
    """
    df_test = df.iloc[test_idx].copy()
    
    # Throughput for each fixed strategy
    always_reno  = df_test["reno_tp"].mean()
    always_cubic = df_test["cubic_tp"].mean()
    always_bbr   = df_test["bbr_tp"].mean()
    
    # Throughput for ML selection
    tp_cols = {"reno": "reno_tp", "cubic": "cubic_tp", "bbr": "bbr_tp"}
    ml_throughputs = []
    for _, row in df_test.iterrows():
        predicted = row.get("predicted_variant", row["best_variant"])
        tp_col = tp_cols.get(predicted, "bbr_tp")
        ml_throughputs.append(row[tp_col])
    ml_adaptive = np.mean(ml_throughputs)
    
    # Oracle (always picks the best – upper bound)
    oracle = df_test["best_throughput"].mean() if "best_throughput" in df_test.columns else ml_adaptive
    
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPARISON (Held-out Test Conditions)")
    print(f"{'='*60}")
    print(f"  Strategy            Avg Throughput (Mbps)  vs Best Fixed")
    print(f"  ─────────────────   ─────────────────────  ─────────────")
    
    best_fixed = max(always_reno, always_cubic, always_bbr)
    strategies = {
        "Always Reno":    always_reno,
        "Always Cubic":   always_cubic,
        "Always BBR":     always_bbr,
        "ML Adaptive":    ml_adaptive,
        "Oracle (ideal)": oracle
    }
    
    for name, tp in strategies.items():
        improvement = ((tp - best_fixed) / best_fixed * 100) if best_fixed > 0 else 0
        marker = " ◄ BEST" if tp == max(strategies.values()) else ""
        print(f"  {name:20s}  {tp:8.2f} Mbps          {improvement:+.1f}%{marker}")
    
    result = {
        "always_reno_tp": round(always_reno, 2),
        "always_cubic_tp": round(always_cubic, 2),
        "always_bbr_tp": round(always_bbr, 2),
        "ml_adaptive_tp": round(ml_adaptive, 2),
        "oracle_tp": round(oracle, 2),
        "improvement_over_best_fixed_pct": round(
            (ml_adaptive - best_fixed) / best_fixed * 100, 2
        ) if best_fixed > 0 else 0
    }
    
    return result


# ---------------------------------------------------------------------------
# Save model and results
# ---------------------------------------------------------------------------
def save_model(model, model_name, model_dir=MODEL_DIR):
    """Save trained model to disk."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    print(f"[SAVED] Model → {path}")
    return path


def save_results(results, results_dir=RESULTS_DIR):
    """Save evaluation results to JSON."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "evaluation_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVED] Results → {path}")
    return path


# ---------------------------------------------------------------------------
# Full ML pipeline
# ---------------------------------------------------------------------------
def run_ml_pipeline(ml_csv=ML_CSV):
    """
    Execute the complete ML pipeline:
    1. Load data
    2. Held-out split
    3. Train Decision Tree baseline
    4. Train Random Forest
    5. Evaluate both models
    6. Run baseline comparison
    7. Save models and results
    """
    print("\n" + "=" * 70)
    print("  Module M5: Machine Learning Pipeline")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Load data
    df = load_ml_data(ml_csv)
    
    # Held-out split
    X_train, X_test, y_train, y_test, train_idx, test_idx = held_out_split(df)
    
    # Train models
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate
    dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree (Baseline)")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Feature importance
    dt_importance = get_feature_importance(dt_model)
    rf_importance = get_feature_importance(rf_model)
    
    # Add ML predictions to dataset for baseline comparison
    df_with_pred = df.copy()
    df_with_pred.loc[df_with_pred.index[test_idx], "predicted_variant"] = rf_model.predict(X_test)
    
    # Baseline comparison
    comparison = baseline_comparison(df_with_pred, test_idx)
    
    # Save everything
    save_model(dt_model, "decision_tree")
    save_model(rf_model, "random_forest")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "dt_feature_importance": dt_importance,
        "rf_feature_importance": rf_importance,
        "baseline_comparison": comparison
    }
    save_results(all_results)
    
    print(f"\n{'='*70}")
    print(f"  ML PIPELINE COMPLETE")
    print(f"  Random Forest Accuracy: {rf_metrics['accuracy']:.2%}")
    print(f"  ML Adaptive Throughput: {comparison['ml_adaptive_tp']:.2f} Mbps")
    print(f"  Improvement over best fixed: "
          f"{comparison['improvement_over_best_fixed_pct']:+.2f}%")
    print(f"{'='*70}")
    
    return all_results, rf_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results, model = run_ml_pipeline()
