import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import sklearn.utils.validation as sk_validation


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

from train_tabpfn_demo import build_demo_feature_table, evaluate


OUT_DIR = Path(r"D:\nn\process\tabpfn_demo_cv")


def run_fold(fold_id: int, train_idx, test_idx, x, y, full_df):
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = TabPFNClassifier(device="cuda", N_ensemble_configurations=16, seed=fold_id + 42)
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    baseline_train_prob = full_df.iloc[train_idx]["best_score"].to_numpy()
    baseline_test_prob = full_df.iloc[test_idx]["best_score"].to_numpy()

    rows = []
    for model_name, split_name, y_true, prob in [
        ("TabPFN_second_stage", "train", y_train, train_prob),
        ("TabPFN_second_stage", "test", y_test, test_prob),
        ("XGB_base_score", "train", y_train, baseline_train_prob),
        ("XGB_base_score", "test", y_test, baseline_test_prob),
    ]:
        rows.append(
            {
                "fold": fold_id,
                "model": model_name,
                "split": split_name,
                **evaluate(y_true, prob, threshold=0.5),
            }
        )
    return rows


def plot_results(results_df):
    sns.set_theme(style="whitegrid")
    test_df = results_df[results_df["split"] == "test"].copy()
    metric_order = ["auc", "precision", "recall", "accuracy"]
    pretty = {"auc": "AUC", "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

    ax = axes[0, 0]
    melted = test_df.melt(
        id_vars=["fold", "model", "split"],
        value_vars=metric_order,
        var_name="metric",
        value_name="value",
    )
    sns.boxplot(data=melted, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("5-fold CV Test Metrics")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_xticklabels([pretty[m] for m in metric_order])
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    sns.lineplot(data=test_df, x="fold", y="auc", hue="model", marker="o", ax=ax)
    ax.set_title("Test AUC Across Folds")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    summary_long = (
        test_df.groupby("model")[metric_order]
        .agg(["mean", "std"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "metric"})
    )
    sns.barplot(data=summary_long, x="metric", y="mean", hue="model", ax=ax)
    for patch, (_, row) in zip(ax.patches, summary_long.iterrows()):
        xpos = patch.get_x() + patch.get_width() / 2
        ax.errorbar(x=xpos, y=row["mean"], yerr=row["std"], color="black", capsize=3, linewidth=1)
    ax.set_title("Mean +/- SD on 5-fold Test Splits")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Value")
    ax.set_xticklabels([pretty[m] for m in metric_order])
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    sns.scatterplot(data=test_df, x="recall", y="precision", hue="model", style="model", s=90, ax=ax)
    ax.set_title("Precision-Recall Tradeoff on CV Test Folds")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")

    fig.suptitle("Stage 2 5-fold Cross-validation on Experimental Data", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cv_5fold_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_df, x, y, _ = build_demo_feature_table()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        all_rows.extend(run_fold(fold_id, train_idx, test_idx, x, y, full_df))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "cv_metrics.csv", index=False)

    summary_df = (
        results_df[results_df["split"] == "test"]
        .groupby("model")[["auc", "precision", "recall", "accuracy"]]
        .agg(["mean", "std", "min", "max"])
    )
    summary_df.to_csv(OUT_DIR / "cv_summary.csv")

    plot_results(results_df)

    report_rows = (
        summary_df.copy()
        .rename_axis(index="model")
        .reset_index()
    )
    report_rows.columns = [
        "model",
        "auc_mean","auc_std","auc_min","auc_max",
        "precision_mean","precision_std","precision_min","precision_max",
        "recall_mean","recall_std","recall_min","recall_max",
        "accuracy_mean","accuracy_std","accuracy_min","accuracy_max",
    ]
    report = {"n_folds": 5, "test_summary_rows": report_rows.to_dict(orient="records")}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(results_df[results_df["split"] == "test"].to_string(index=False))
    print(summary_df.to_string())
    print(f"saved={OUT_DIR / 'cv_5fold_overview.png'}")


if __name__ == "__main__":
    main()
