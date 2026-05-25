import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


TRAIN_PATH = Path(r"D:\nn\process\bio_features_full\train_features.csv")
VAL_PATH = Path(r"D:\nn\process\bio_features_full\val_features.csv")
TEST_PATH = Path(r"D:\nn\process\bio_features_full\test_features.csv")
RESULT_DIR = Path(r"D:\nn\process\xgb_results")


def load_split(path: Path):
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col not in {"mirna", "gene", "label"}]
    x = df[feature_cols].astype(np.float32)
    y = df["label"].astype(int).to_numpy()
    return df, x, y, feature_cols


def roc_auc(y_true, prob):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, prob))


def load_best_params():
    metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    return metrics["chosen_params"]


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, x_train, y_train, feature_cols = load_split(TRAIN_PATH)
    _, x_val, y_val, val_cols = load_split(VAL_PATH)
    _, x_test, y_test, test_cols = load_split(TEST_PATH)

    if feature_cols != val_cols or feature_cols != test_cols:
        raise ValueError("Feature columns are inconsistent across splits.")

    feature_audit = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "dtype": [str(x_train[col].dtype) for col in feature_cols],
        }
    )
    feature_audit.to_csv(RESULT_DIR / "feature_columns_audit.csv", index=False)

    params = load_best_params()
    rng = np.random.default_rng(20260415)
    shuffled_y_train = rng.permutation(y_train)

    model = XGBClassifier(**params)
    model.fit(x_train, shuffled_y_train, eval_set=[(x_val, y_val)], verbose=False)

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    summary = {
        "feature_count": len(feature_cols),
        "contains_label_as_feature": "label" in feature_cols,
        "contains_gene_as_feature": "gene" in feature_cols,
        "contains_mirna_as_feature": "mirna" in feature_cols,
        "shuffled_train_auc": roc_auc(y_train, train_prob),
        "shuffled_val_auc": roc_auc(y_val, val_prob),
        "shuffled_test_auc": roc_auc(y_test, test_prob),
    }

    (RESULT_DIR / "label_leakage_audit.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("top_feature_columns")
    print(feature_audit.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
