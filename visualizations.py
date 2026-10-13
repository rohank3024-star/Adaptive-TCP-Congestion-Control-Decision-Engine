#!/usr/bin/env python3
"""
Visualization Module
=====================
Generates all charts and plots for the project:
  1. Throughput comparison by TCP variant (bar chart)
  2. Throughput heatmap per variant across conditions
  3. Best variant distribution (pie chart)
  4. Feature importance (horizontal bar chart)
  5. Confusion matrix (heatmap)
  6. Baseline comparison (bar chart)
  7. Throughput vs delay/loss/bandwidth line plots
  8. Best variant heatmap (delay vs loss grid)

All plots are saved to the 'plots/' directory.

Author: Shreyans Bohra (24BCT0013) – Data & ML
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLOT_DIR    = "plots"
ML_CSV      = "data/dataset_b_ml.csv"
RAW_CSV     = "data/dataset_a_raw.csv"
RESULTS_JSON = "results/evaluation_results.json"

# Color palette
COLORS = {
    "reno":  "#FF6B6B",   # Coral red
    "cubic": "#4ECDC4",   # Teal
    "bbr":   "#45B7D1",   # Sky blue
    "ml":    "#96CEB4",   # Sage green
    "oracle": "#FFEAA7",  # Pale yellow
}

# Style
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#ccc",
    "ytick.color": "#ccc",
    "grid.color": "#333",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
})

sns.set_palette([COLORS["reno"], COLORS["cubic"], COLORS["bbr"]])


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Throughput comparison by variant
# ---------------------------------------------------------------------------
def plot_throughput_comparison(df_raw, save=True):
    """Box plot comparing throughput distributions across TCP variants."""
    ensure_plot_dir()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = ["reno", "cubic", "bbr"]
    data = [df_raw[df_raw["tcp_variant"] == v]["throughput_mbps"] for v in variants]
    
    bp = ax.boxplot(data, labels=["TCP Reno", "TCP Cubic", "TCP BBR"],
                    patch_artist=True, notch=True, widths=0.5)
    
    for patch, color in zip(bp["boxes"], [COLORS["reno"], COLORS["cubic"], COLORS["bbr"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#eee")
    bp["medians"][0].set_color("#fff")
    bp["medians"][1].set_color("#fff")
    bp["medians"][2].set_color("#fff")
    
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("TCP Throughput Distribution by Variant")
    ax.grid(True, axis="y")
    
    # Add median annotations
    medians = [np.median(d) for d in data]
    for i, med in enumerate(medians):
        ax.text(i + 1, med + 1, f"{med:.1f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10, color="#fff")
    
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "01_throughput_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Best variant distribution
# ---------------------------------------------------------------------------
def plot_best_variant_distribution(df_ml, save=True):
    """Pie chart showing distribution of best-performing TCP variant."""
    ensure_plot_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    counts = df_ml["best_variant"].value_counts()
    colors_list = [COLORS.get(v, "#888") for v in counts.index]
    
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=counts.index.str.upper(),
        colors=colors_list, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2}
    )
    for autotext in autotexts:
        autotext.set_color("#1a1a2e")
        autotext.set_fontweight("bold")
    
    ax1.set_title("Best TCP Variant Distribution\n(Across All Network Conditions)")
    
    # Bar chart breakdown by bandwidth
    variant_by_bw = df_ml.groupby(["bandwidth_mbps", "best_variant"]).size().unstack(fill_value=0)
    variant_by_bw.plot(kind="bar", stacked=True, ax=ax2,
                       color=[COLORS.get(c, "#888") for c in variant_by_bw.columns],
                       edgecolor="#1a1a2e")
    ax2.set_xlabel("Bandwidth (Mbps)")
    ax2.set_ylabel("Number of Conditions")
    ax2.set_title("Best Variant by Bandwidth")
    ax2.legend(title="TCP Variant", loc="upper left")
    ax2.tick_params(axis="x", rotation=0)
    
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "02_best_variant_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Throughput vs network parameters
# ---------------------------------------------------------------------------
def plot_throughput_vs_params(df_raw, save=True):
    """Line plots: throughput vs bandwidth, delay, and loss for each variant."""
    ensure_plot_dir()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    params = [
        ("bandwidth_mbps", "Bandwidth (Mbps)", axes[0]),
        ("delay_ms", "Delay (ms)", axes[1]),
        ("loss_pct", "Packet Loss (%)", axes[2]),
    ]
    
    for param, label, ax in params:
        for variant in ["reno", "cubic", "bbr"]:
            df_v = df_raw[df_raw["tcp_variant"] == variant]
            grouped = df_v.groupby(param)["throughput_mbps"].median()
            ax.plot(grouped.index, grouped.values,
                    marker="o", linewidth=2, markersize=6,
                    label=variant.upper(), color=COLORS[variant])
        
        ax.set_xlabel(label)
        ax.set_ylabel("Median Throughput (Mbps)")
        ax.set_title(f"Throughput vs {label}")
        ax.legend()
        ax.grid(True)
    
    plt.suptitle("TCP Variant Performance Under Varying Conditions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "03_throughput_vs_params.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Best variant heatmap (delay vs loss)
# ---------------------------------------------------------------------------
def plot_variant_heatmap(df_ml, save=True):
    """Heatmap showing best variant across delay × loss grid for different bandwidths."""
    ensure_plot_dir()
    
    # Select representative bandwidths
    bandwidths = sorted(df_ml["bandwidth_mbps"].unique())
    n_bw = min(6, len(bandwidths))
    selected_bw = bandwidths[:n_bw]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    variant_map = {"reno": 0, "cubic": 1, "bbr": 2}
    cmap = matplotlib.colors.ListedColormap([COLORS["reno"], COLORS["cubic"], COLORS["bbr"]])
    
    for idx, bw in enumerate(selected_bw):
        if idx >= 6:
            break
        ax = axes[idx]
        df_bw = df_ml[df_ml["bandwidth_mbps"] == bw]
        
        pivot = df_bw.pivot_table(
            index="delay_ms", columns="loss_pct",
            values="best_variant", aggfunc="first"
        )
        
        # Convert to numeric for heatmap
        pivot_num = pivot.map(lambda x: variant_map.get(x, -1) if pd.notna(x) else -1)
        
        sns.heatmap(pivot_num, ax=ax, cmap=cmap, vmin=0, vmax=2,
                    annot=pivot, fmt="", cbar=False,
                    linewidths=1, linecolor="#1a1a2e",
                    annot_kws={"fontsize": 9, "fontweight": "bold"})
        
        ax.set_title(f"BW = {bw} Mbps")
        ax.set_xlabel("Packet Loss (%)")
        ax.set_ylabel("Delay (ms)")
    
    # Add legend
    patches = [mpatches.Patch(color=COLORS[v], label=v.upper()) for v in ["reno", "cubic", "bbr"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=12,
              framealpha=0.8, edgecolor="#e94560")
    
    plt.suptitle("Best TCP Variant by Network Conditions",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if save:
        path = os.path.join(PLOT_DIR, "04_variant_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(results, save=True):
    """Horizontal bar chart of Random Forest feature importance."""
    ensure_plot_dir()
    
    importance = results.get("rf_feature_importance", {})
    if not importance:
        print("[SKIP] No feature importance data found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    features = list(importance.keys())
    values = list(importance.values())
    
    # Sort by importance
    sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
    features, values = zip(*sorted_pairs)
    
    colors = ["#e94560" if v == max(values) else "#45B7D1" for v in values]
    
    bars = ax.barh(features, values, color=colors, edgecolor="#1a1a2e", height=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontweight="bold", fontsize=11)
    
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest – Feature Importance")
    ax.set_xlim(0, max(values) * 1.2)
    
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "05_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(results, save=True):
    """Heatmap confusion matrix for the Random Forest classifier."""
    ensure_plot_dir()
    
    rf = results.get("random_forest", {})
    cm = rf.get("confusion_matrix", None)
    if cm is None:
        print("[SKIP] No confusion matrix data found.")
        return
    
    cm = np.array(cm)
    labels = ["BBR", "Cubic", "Reno"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="#1a1a2e", ax=ax1,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix (Counts)")
    
    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="#1a1a2e", ax=ax2,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (Normalized)")
    
    plt.suptitle(f"Random Forest Classification Results "
                 f"(Accuracy: {rf.get('accuracy', 'N/A')})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "06_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Baseline comparison chart
# ---------------------------------------------------------------------------
def plot_baseline_comparison(results, save=True):
    """Bar chart comparing fixed TCP strategies vs ML adaptive selection."""
    ensure_plot_dir()
    
    comp = results.get("baseline_comparison", {})
    if not comp:
        print("[SKIP] No baseline comparison data found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = {
        "Always\nReno":   comp.get("always_reno_tp", 0),
        "Always\nCubic":  comp.get("always_cubic_tp", 0),
        "Always\nBBR":    comp.get("always_bbr_tp", 0),
        "ML\nAdaptive":   comp.get("ml_adaptive_tp", 0),
        "Oracle\n(Ideal)": comp.get("oracle_tp", 0),
    }
    
    names = list(strategies.keys())
    values = list(strategies.values())
    colors_list = [COLORS["reno"], COLORS["cubic"], COLORS["bbr"],
                   COLORS["ml"], COLORS["oracle"]]
    
    bars = ax.bar(names, values, color=colors_list, edgecolor="#1a1a2e",
                  width=0.6, linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom",
                fontweight="bold", fontsize=12, color="#eee")
    
    # Highlight ML bar
    bars[3].set_edgecolor("#FFD700")
    bars[3].set_linewidth(3)
    
    ax.set_ylabel("Average Throughput (Mbps)")
    ax.set_title("Baseline Comparison: Fixed TCP vs ML Adaptive Selection\n"
                 f"(ML improvement: {comp.get('improvement_over_best_fixed_pct', 0):+.1f}% "
                 f"over best fixed strategy)")
    ax.grid(True, axis="y")
    
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "07_baseline_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Retransmissions comparison
# ---------------------------------------------------------------------------
def plot_retransmissions(df_raw, save=True):
    """Box plot comparing retransmissions across TCP variants."""
    ensure_plot_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # By variant
    for variant in ["reno", "cubic", "bbr"]:
        df_v = df_raw[df_raw["tcp_variant"] == variant]
        grouped = df_v.groupby("loss_pct")["retransmissions"].median()
        ax1.plot(grouped.index, grouped.values,
                marker="s", linewidth=2, markersize=7,
                label=variant.upper(), color=COLORS[variant])
    
    ax1.set_xlabel("Packet Loss (%)")
    ax1.set_ylabel("Median Retransmissions")
    ax1.set_title("Retransmissions vs Packet Loss")
    ax1.legend()
    ax1.grid(True)
    
    # By variant and bandwidth
    for variant in ["reno", "cubic", "bbr"]:
        df_v = df_raw[df_raw["tcp_variant"] == variant]
        grouped = df_v.groupby("bandwidth_mbps")["retransmissions"].median()
        ax2.plot(grouped.index, grouped.values,
                marker="o", linewidth=2, markersize=7,
                label=variant.upper(), color=COLORS[variant])
    
    ax2.set_xlabel("Bandwidth (Mbps)")
    ax2.set_ylabel("Median Retransmissions")
    ax2.set_title("Retransmissions vs Bandwidth")
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle("TCP Retransmission Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "08_retransmissions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 9. Model comparison (DT vs RF)
# ---------------------------------------------------------------------------
def plot_model_comparison(results, save=True):
    """Compare Decision Tree and Random Forest metrics."""
    ensure_plot_dir()
    
    dt = results.get("decision_tree", {})
    rf = results.get("random_forest", {})
    
    if not dt or not rf:
        print("[SKIP] Need both DT and RF results.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    dt_vals = [dt.get(m, 0) for m in metrics]
    rf_vals = [rf.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.3
    
    bars1 = ax.bar(x - width/2, dt_vals, width, label="Decision Tree",
                   color="#FF6B6B", edgecolor="#1a1a2e", alpha=0.8)
    bars2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest",
                   color="#45B7D1", edgecolor="#1a1a2e", alpha=0.8)
    
    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontweight="bold", fontsize=10)
    
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Decision Tree vs Random Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score"])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y")
    
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "09_model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------
def generate_all_plots():
    """Generate all visualization plots."""
    print("=" * 60)
    print("  Generating All Visualizations")
    print("=" * 60)
    
    # Load data
    df_raw = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else None
    df_ml  = pd.read_csv(ML_CSV) if os.path.exists(ML_CSV) else None
    
    results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            results = json.load(f)
    
    if df_raw is not None:
        plot_throughput_comparison(df_raw)
        plot_throughput_vs_params(df_raw)
        plot_retransmissions(df_raw)
    else:
        print("[SKIP] Raw CSV not found – skipping raw data plots.")
    
    if df_ml is not None:
        plot_best_variant_distribution(df_ml)
        plot_variant_heatmap(df_ml)
    else:
        print("[SKIP] ML CSV not found – skipping ML data plots.")
    
    if results:
        plot_feature_importance(results)
        plot_confusion_matrix(results)
        plot_baseline_comparison(results)
        plot_model_comparison(results)
    else:
        print("[SKIP] Results JSON not found – skipping evaluation plots.")
    
    print(f"\n[DONE] All plots saved to {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_all_plots()
