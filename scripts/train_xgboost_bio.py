import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


def load_split(path: Path):
    df = pd.read_csv(path)
    drop_cols = ["mirna", "gene", "label"]
    x = df.drop(columns=drop_cols).astype(np.float32)
    y = df["label"].astype(int).to_numpy()
    return df, x, y


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "positive_rate": float(pred.mean()),
    }


def choose_threshold(y_true, prob, min_recall=0.85):
    candidates = np.unique(np.round(prob, 6))
    candidates = np.concatenate(([0.0], candidates, [1.0]))

    feasible = []
    fallback = []
    for thr in candidates:
        metrics = evaluate(y_true, prob, float(thr))
        row = {"threshold": float(thr), **metrics}
        fallback.append(row)
        if metrics["recall"] >= min_recall:
            feasible.append(row)

    if feasible:
        feasible.sort(
            key=lambda r: (
                r["precision"],
                r["accuracy"],
                -r["positive_rate"],
                r["threshold"],
            ),
            reverse=True,
        )
        return feasible[0], pd.DataFrame(feasible).sort_values("threshold").reset_index(drop=True)

    fallback.sort(
        key=lambda r: (
            r["recall"],
            r["precision"],
            r["accuracy"],
            -r["positive_rate"],
            r["threshold"],
        ),
        reverse=True,
    )
    return fallback[0], pd.DataFrame(fallback).sort_values("threshold").reset_index(drop=True)


def build_models(pos_weight):
    common = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=80,
        random_state=42,
        n_jobs=0,
        scale_pos_weight=pos_weight,
    )
    return [
        {
            "name": "model_a",
            "params": {
                **common,
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 2,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "gamma": 0.0,
                "max_bin": 256,
            },
        },
        {
            "name": "model_b",
            "params": {
                **common,
                "n_estimators": 700,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "reg_lambda": 1.5,
                "reg_alpha": 0.0,
                "gamma": 0.0,
                "max_bin": 256,
            },
        },
        {
            "name": "model_c",
            "params": {
                **common,
                "n_estimators": 350,
                "max_depth": 3,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 3,
                "reg_lambda": 2.0,
                "reg_alpha": 0.0,
                "gamma": 0.0,
                "max_bin": 512,
            },
        },
        {
            "name": "model_d",
            "params": {
                **common,
                "n_estimators": 900,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "min_child_weight": 1,
                "reg_lambda": 1.0,
                "reg_alpha": 0.1,
                "gamma": 0.0,
                "max_bin": 256,
            },
        },
        {
            "name": "model_e",
            "params": {
                **common,
                "n_estimators": 1200,
                "max_depth": 8,
                "learning_rate": 0.025,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "reg_lambda": 2.0,
                "reg_alpha": 0.2,
                "gamma": 0.0,
                "max_bin": 256,
            },
        },
        {
            "name": "model_f",
            "params": {
                **common,
                "n_estimators": 650,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 1.0,
                "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "reg_lambda": 3.0,
                "reg_alpha": 0.0,
                "gamma": 0.0,
                "max_bin": 512,
            },
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=r"D:\nn\process\bio_features_full\train_features.csv")
    parser.add_argument("--val", default=r"D:\nn\process\bio_features_full\val_features.csv")
    parser.add_argument("--test", default=r"D:\nn\process\bio_features_full\test_features.csv")
    parser.add_argument("--out-dir", default=r"D:\nn\process\xgb_results")
    parser.add_argument("--min-recall", type=float, default=0.85)
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, x_train, y_train = load_split(train_path)
    val_df, x_val, y_val = load_split(val_path)
    test_df, x_test, y_test = load_split(test_path)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    leaderboard = []
    best_artifact = None

    for spec in build_models(pos_weight):
        model = XGBClassifier(**spec["params"])
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        train_prob = model.predict_proba(x_train)[:, 1]
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]

        threshold_row, threshold_table = choose_threshold(y_val, val_prob, min_recall=args.min_recall)
        threshold = threshold_row["threshold"]

        train_metrics = evaluate(y_train, train_prob, threshold)
        val_metrics = evaluate(y_val, val_prob, threshold)
        test_metrics = evaluate(y_test, test_prob, threshold)

        record = {
            "model": spec["name"],
            "threshold": threshold,
            "train_auc": train_metrics["auc"],
            "val_auc": val_metrics["auc"],
            "test_auc": test_metrics["auc"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_accuracy": val_metrics["accuracy"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_accuracy": test_metrics["accuracy"],
        }
        leaderboard.append(record)

        score = (
            val_metrics["auc"] >= 0.9,
            val_metrics["recall"] >= 0.85,
            val_metrics["auc"],
            val_metrics["precision"],
            val_metrics["accuracy"],
        )
        if best_artifact is None or score > best_artifact["score"]:
            best_artifact = {
                "score": score,
                "spec": spec,
                "model": model,
                "threshold": threshold,
                "threshold_table": threshold_table,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "train_prob": train_prob,
                "val_prob": val_prob,
                "test_prob": test_prob,
            }

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        ["val_auc", "val_precision", "val_accuracy"], ascending=False
    )
    leaderboard_df.to_csv(out_dir / "leaderboard.csv", index=False)

    booster = best_artifact["model"].get_booster()
    booster.save_model(str(out_dir / "best_model.json"))

    feature_importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance_gain": best_artifact["model"].feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

    best_artifact["threshold_table"].to_csv(out_dir / "val_threshold_scan.csv", index=False)

    metrics_payload = {
        "chosen_model": best_artifact["spec"]["name"],
        "chosen_params": best_artifact["spec"]["params"],
        "chosen_threshold": best_artifact["threshold"],
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "min_recall_target": args.min_recall,
        "train_metrics": best_artifact["train_metrics"],
        "val_metrics": best_artifact["val_metrics"],
        "test_metrics": best_artifact["test_metrics"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
