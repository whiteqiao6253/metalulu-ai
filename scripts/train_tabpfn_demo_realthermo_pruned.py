import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.utils.validation as sk_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


if "force_all_finite" not in inspect.signature(sk_validation.check_X_y).parameters:
    _original_check_x_y = sk_validation.check_X_y

    def _compat_check_x_y(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return _original_check_x_y(*args, **kwargs)

    sk_validation.check_X_y = _compat_check_x_y

if "force_all_finite" not in inspect.signature(sk_validation.check_array).parameters:
    _original_check_array = sk_validation.check_array

    def _compat_check_array(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return _original_check_array(*args, **kwargs)

    sk_validation.check_array = _compat_check_array

from tabpfn import TabPFNClassifier

from train_tabpfn_demo_realthermo import build_realthermo_feature_table


OUT_DIR = Path(r"D:\nn\process\tabpfn_demo_realthermo_pruned_results")

DROP_REALTHERMO_REDUNDANT_FEATURES = {
    "max_score",
    "mean_score",
    "min_score",
    "score",
    "site_type_weight",
}


def evaluate(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "positive_rate": float(pred.mean()),
    }


def load_realthermo_pruned_features():
    full_df, x, y, feature_cols = build_realthermo_feature_table()
    keep_cols = [col for col in feature_cols if col not in DROP_REALTHERMO_REDUNDANT_FEATURES]
    return full_df, x[keep_cols].copy(), y, keep_cols


def run_single_split(full_df, x, y):
    indices = np.arange(len(full_df))
    train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)

    model = TabPFNClassifier(device="cuda", N_ensemble_configurations=16, seed=42)
    model.fit(x.iloc[train_idx], y[train_idx])

    train_prob = model.predict_proba(x.iloc[train_idx])[:, 1]
    test_prob = model.predict_proba(x.iloc[test_idx])[:, 1]
    base_train_prob = full_df.iloc[train_idx]["best_score"].to_numpy()
    base_test_prob = full_df.iloc[test_idx]["best_score"].to_numpy()

    train_scored = full_df.iloc[train_idx][["mirna_seq", "gene_name", "label", "best_score"]].copy()
    train_scored["tabpfn_realthermo_pruned_score"] = train_prob
    train_scored["tabpfn_realthermo_pruned_pred_label"] = (train_prob >= 0.5).astype(int)
    train_scored["correct_tabpfn_realthermo_pruned"] = (
        train_scored["tabpfn_realthermo_pruned_pred_label"] == train_scored["label"]
    ).astype(int)
    train_scored.to_csv(OUT_DIR / "demo_train_scores_realthermo_pruned.csv", index=False)

    test_scored = full_df.iloc[test_idx][["mirna_seq", "gene_name", "label", "best_score"]].copy()
    test_scored["tabpfn_realthermo_pruned_score"] = test_prob
    test_scored["tabpfn_realthermo_pruned_pred_label"] = (test_prob >= 0.5).astype(int)
    test_scored["xgb_pred_label"] = (test_scored["best_score"].to_numpy() >= 0.5).astype(int)
    test_scored["correct_tabpfn_realthermo_pruned"] = (
        test_scored["tabpfn_realthermo_pruned_pred_label"] == test_scored["label"]
    ).astype(int)
    test_scored = test_scored.sort_values("tabpfn_realthermo_pruned_score", ascending=False)
    test_scored.to_csv(OUT_DIR / "demo_test_scores_realthermo_pruned.csv", index=False)

    return [
        {
            "setting": "single_70_30",
            "split": "train",
            "model": "RealThermo_Pruned_TabPFN",
            **evaluate(y[train_idx], train_prob),
        },
        {
            "setting": "single_70_30",
            "split": "test",
            "model": "RealThermo_Pruned_TabPFN",
            **evaluate(y[test_idx], test_prob),
        },
        {"setting": "single_70_30", "split": "train", "model": "XGB_base_score", **evaluate(y[train_idx], base_train_prob)},
        {"setting": "single_70_30", "split": "test", "model": "XGB_base_score", **evaluate(y[test_idx], base_test_prob)},
    ]


def run_repeated(full_df, x, y, n_repeats=10):
    rows = []
    indices = np.arange(len(full_df))
    for seed in range(n_repeats):
        train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=y)
        model = TabPFNClassifier(device="cuda", N_ensemble_configurations=16, seed=seed)
        model.fit(x.iloc[train_idx], y[train_idx])
        test_prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        base_test_prob = full_df.iloc[test_idx]["best_score"].to_numpy()
        rows.append(
            {
                "setting": "repeated_70_30",
                "split": "test",
                "repeat": seed,
                "model": "RealThermo_Pruned_TabPFN",
                **evaluate(y[test_idx], test_prob),
            }
        )
        rows.append(
            {
                "setting": "repeated_70_30",
                "split": "test",
                "repeat": seed,
                "model": "XGB_base_score",
                **evaluate(y[test_idx], base_test_prob),
            }
        )
    return rows


def run_cv(full_df, x, y, n_splits=5):
    rows = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        model = TabPFNClassifier(device="cuda", N_ensemble_configurations=16, seed=fold)
        model.fit(x.iloc[train_idx], y[train_idx])
        test_prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        base_test_prob = full_df.iloc[test_idx]["best_score"].to_numpy()
        rows.append(
            {
                "setting": "cv_5fold",
                "split": "test",
                "fold": fold,
                "model": "RealThermo_Pruned_TabPFN",
                **evaluate(y[test_idx], test_prob),
            }
        )
        rows.append(
            {
                "setting": "cv_5fold",
                "split": "test",
                "fold": fold,
                "model": "XGB_base_score",
                **evaluate(y[test_idx], base_test_prob),
            }
        )
    return rows


def summarize(df):
    test_df = df[df["split"] == "test"].copy()
    summary = (
        test_df.groupby(["setting", "model"])[["auc", "precision", "recall", "accuracy"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def plot_results(summary_df):
    sns.set_theme(style="whitegrid")
    metric_cols = {
        "auc_mean": "AUC",
        "precision_mean": "Precision",
        "recall_mean": "Recall",
        "accuracy_mean": "Accuracy",
    }
    plot_df = summary_df.melt(
        id_vars=["setting", "model"],
        value_vars=list(metric_cols.keys()),
        var_name="metric",
        value_name="value",
    )
    plot_df["metric"] = plot_df["metric"].map(metric_cols)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, setting in zip(axes, ["single_70_30", "repeated_70_30", "cv_5fold"]):
        subset = plot_df[plot_df["setting"] == setting]
        sns.barplot(data=subset, x="metric", y="value", hue="model", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title(setting.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylabel("Value" if ax is axes[0] else "")
        for patch in ax.patches:
            height = patch.get_height()
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )
    fig.suptitle("Real-thermodynamic Pruned Second-layer TabPFN", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "realthermo_pruned_stage2_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_df, x, y, keep_cols = load_realthermo_pruned_features()

    pd.DataFrame({"feature_name": keep_cols}).to_csv(OUT_DIR / "feature_columns_realthermo_pruned.csv", index=False)
    pd.DataFrame({"dropped_feature": sorted(DROP_REALTHERMO_REDUNDANT_FEATURES)}).to_csv(
        OUT_DIR / "dropped_realthermo_redundant_features.csv",
        index=False,
    )

    rows = []
    rows.extend(run_single_split(full_df, x, y))
    rows.extend(run_repeated(full_df, x, y))
    rows.extend(run_cv(full_df, x, y))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metrics_realthermo_pruned_all.csv", index=False)
    summary_df = summarize(metrics_df)
    summary_df.to_csv(OUT_DIR / "metrics_realthermo_pruned_summary.csv", index=False)
    plot_results(summary_df)

    report = {
        "n_rows": int(len(full_df)),
        "feature_count": int(len(keep_cols)),
        "dropped_features": sorted(DROP_REALTHERMO_REDUNDANT_FEATURES),
        "test_summary": summary_df.to_dict(orient="records"),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
