"""
Deep SHAP interpretability for the 90-feature XGBoost model.
Generates:
  1. SHAP dependence plots for top features
  2. SHAP waterfall plots for individual human/AI cases
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "processed"


def load_data_and_model():
    """Load cached 90-feature data and retrain XGBoost (fast on CPU features)."""
    cache = DATA_DIR / "hc3_extended_features.csv"
    if not cache.exists():
        print("ERROR: Run run_extended.py first to generate features.")
        sys.exit(1)

    feat_df = pd.read_csv(cache)
    numeric_cols = [c for c in feat_df.columns if c not in {"label", "label_name", "source"}]
    all_feature_cols = numeric_cols

    X = feat_df[all_feature_cols]
    y = feat_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    clf.fit(X_train, y_train, verbose=False)
    print(f"XGBoost retrained: {X_train.shape[1]} features, "
          f"AUC={__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]):.4f}")

    return clf, X_train, X_test, y_train, y_test, feat_df


def compute_shap(clf, X_train):
    """Compute SHAP values on a subsample."""
    sample_idx = np.random.RandomState(42).choice(len(X_train), min(2000, len(X_train)), replace=False)
    X_sample = X_train.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_sample)
    return shap_values, X_sample


def plot_dependence(shap_values, X_sample):
    """SHAP dependence plots for top 6 features."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:6]
    top_features = [X_sample.columns[i] for i in top_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        feat_idx = list(X_sample.columns).index(feat)
        feat_vals = X_sample[feat].values
        shap_vals = shap_values.values[:, feat_idx]

        # Find best interaction feature (highest correlation with SHAP residuals)
        best_interact = None
        best_corr = 0
        for j, other in enumerate(X_sample.columns):
            if j == feat_idx:
                continue
            c = abs(np.corrcoef(X_sample[other].values, shap_vals)[0, 1])
            if not np.isnan(c) and c > best_corr:
                best_corr = c
                best_interact = other

        scatter = ax.scatter(feat_vals, shap_vals, c=X_sample[best_interact].values if best_interact else shap_vals,
                             cmap="coolwarm", alpha=0.5, s=8, edgecolors="none")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        if best_interact:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(best_interact, fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        ax.set_title(f"#{i+1}: {feat}", fontsize=11, fontweight="bold")

    fig.suptitle("SHAP Dependence Plots: Top 6 Features (90-feature XGBoost)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = FIG_DIR / "shap_dependence_90feat.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_waterfall(clf, X_test, y_test, feat_df):
    """SHAP waterfall plots for 4 representative cases."""
    explainer = shap.TreeExplainer(clf)

    # Find 4 cases: correct human, correct AI, misclassified human (FP), misclassified AI (FN)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    y_true = y_test.values

    cases = {}
    # Correct human: true=0, pred=0, most confident
    mask = (y_true == 0) & (y_pred == 0)
    if mask.any():
        idx = X_test.index[mask][np.argmin(y_prob[mask])]
        cases["Correct Human (most confident)"] = idx

    # Correct AI: true=1, pred=1, most confident
    mask = (y_true == 1) & (y_pred == 1)
    if mask.any():
        idx = X_test.index[mask][np.argmax(y_prob[mask])]
        cases["Correct AI (most confident)"] = idx

    # False positive: true=0, pred=1
    mask = (y_true == 0) & (y_pred == 1)
    if mask.any():
        idx = X_test.index[mask][np.argmax(y_prob[mask])]
        cases["False Positive (human→AI)"] = idx

    # False negative: true=1, pred=0
    mask = (y_true == 1) & (y_pred == 0)
    if mask.any():
        idx = X_test.index[mask][np.argmin(y_prob[mask])]
        cases["False Negative (AI→human)"] = idx

    n_cases = len(cases)
    if n_cases == 0:
        print("No cases found for waterfall plots.")
        return

    fig, axes = plt.subplots(n_cases, 1, figsize=(14, 5 * n_cases))
    if n_cases == 1:
        axes = [axes]

    for i, (title, idx) in enumerate(cases.items()):
        row = X_test.loc[[idx]]
        sv = explainer(row)

        ax = axes[i]
        # Manual waterfall: top 10 features by |SHAP|
        vals = sv.values[0]
        base = sv.base_values[0]
        if isinstance(base, np.ndarray):
            base = base[1]
        feat_names = list(X_test.columns)

        top_k = 10
        abs_vals = np.abs(vals)
        top_idx = np.argsort(abs_vals)[::-1][:top_k]
        other_val = vals.sum() - vals[top_idx].sum()

        # Build bars
        labels = [feat_names[j] for j in top_idx] + ["other features"]
        bar_vals = [vals[j] for j in top_idx] + [other_val]

        # Draw horizontal waterfall
        cumsum = base
        colors = []
        starts = []
        for v in bar_vals:
            starts.append(cumsum)
            cumsum += v
            colors.append("#d64541" if v > 0 else "#2e86c1")

        y_pos = np.arange(len(labels))[::-1]
        ax.barh(y_pos, bar_vals, left=starts, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

        # Add value labels
        for j, (s, v) in enumerate(zip(starts, bar_vals)):
            x_text = s + v / 2
            ax.text(x_text, y_pos[j], f"{v:+.3f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        # Feature value annotations
        for j, idx_f in enumerate(top_idx):
            feat_val = row.iloc[0, idx_f]
            ax.text(-0.02, y_pos[j], f"= {feat_val:.2f}", ha="right", va="center", fontsize=7,
                    color="gray", transform=ax.get_yaxis_transform())

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(base, color="gray", linewidth=1, linestyle="--", label=f"base={base:.3f}")
        ax.axvline(cumsum, color="black", linewidth=1.5, linestyle="-", label=f"f(x)={cumsum:.3f}")
        ax.set_xlabel("SHAP value (log-odds)", fontsize=10)

        prob = 1 / (1 + np.exp(-cumsum))
        true_label = "Human" if y_true[X_test.index.get_loc(idx)] == 0 else "AI"
        ax.set_title(f"{title} | P(AI)={prob:.4f} | True={true_label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("SHAP Waterfall: How 90 Features Drive Individual Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = FIG_DIR / "shap_waterfall_90feat.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    # Print case details
    for title, idx in cases.items():
        prob = y_prob[X_test.index.get_loc(idx)]
        true = "Human" if y_true[X_test.index.get_loc(idx)] == 0 else "AI"
        print(f"  {title}: idx={idx}, P(AI)={prob:.4f}, true={true}")


if __name__ == "__main__":
    print("Loading data and training XGBoost...")
    clf, X_train, X_test, y_train, y_test, feat_df = load_data_and_model()

    print("\nComputing SHAP values (2000 samples)...")
    shap_values, X_sample = compute_shap(clf, X_train)

    print("\n1. SHAP Dependence Plots (top 6 features)...")
    plot_dependence(shap_values, X_sample)

    print("\n2. SHAP Waterfall Plots (4 representative cases)...")
    plot_waterfall(clf, X_test, y_test, feat_df)

    print("\nDone!")
