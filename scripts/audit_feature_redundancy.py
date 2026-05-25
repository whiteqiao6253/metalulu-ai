import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_tabpfn_demo import build_demo_feature_table


ROOT = Path(r"D:\nn\process")
OUT_DIR = ROOT / "redundancy_audit"
STAGE1_PATH = ROOT / "bio_features_full" / "train_features.csv"


def numeric_features(df: pd.DataFrame, drop_cols: set[str]) -> pd.DataFrame:
    cols = [col for col in df.columns if col not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).astype(np.float64)


def constant_features(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(x)
    for col in x.columns:
        values = x[col]
        nunique = int(values.nunique(dropna=False))
        top_freq = float(values.value_counts(dropna=False, normalize=True).iloc[0])
        std = float(values.std(ddof=0))
        if nunique <= 1 or top_freq >= 0.995 or std == 0.0:
            rows.append(
                {
                    "feature": col,
                    "nunique": nunique,
                    "top_value_frequency": top_freq,
                    "std": std,
                    "n_rows": n,
                }
            )
    columns = ["feature", "nunique", "top_value_frequency", "std", "n_rows"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["top_value_frequency", "nunique"],
        ascending=[False, True],
    )


def exact_duplicate_features(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen = {}
    for col in x.columns:
        key = tuple(pd.util.hash_pandas_object(x[col], index=False).to_numpy())
        if key in seen:
            rows.append({"feature_a": seen[key], "feature_b": col})
        else:
            seen[key] = col
    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b"])
    return pd.DataFrame(rows)


def high_correlation_pairs(x: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr = x.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_pearson_corr"})
    )
    pairs = pairs[pairs["abs_pearson_corr"] >= threshold].copy()
    return pairs.sort_values("abs_pearson_corr", ascending=False)


def audit_matrix(name: str, x: pd.DataFrame) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    constants = constant_features(x)
    duplicates = exact_duplicate_features(x)
    high_corr = high_correlation_pairs(x)

    constants.to_csv(OUT_DIR / f"{name}_constant_or_near_constant.csv", index=False)
    duplicates.to_csv(OUT_DIR / f"{name}_exact_duplicate_features.csv", index=False)
    high_corr.to_csv(OUT_DIR / f"{name}_high_corr_pairs.csv", index=False)

    return {
        "name": name,
        "n_rows": int(len(x)),
        "n_features": int(x.shape[1]),
        "constant_or_near_constant_count": int(len(constants)),
        "exact_duplicate_pair_count": int(len(duplicates)),
        "high_corr_pair_count_abs_ge_0_95": int(len(high_corr)),
        "top_high_corr_pairs": high_corr.head(20).to_dict(orient="records"),
        "constant_or_near_constant_features": constants["feature"].head(50).tolist()
        if not constants.empty
        else [],
    }


def main():
    stage1_df = pd.read_csv(STAGE1_PATH)
    stage1_x = numeric_features(stage1_df, {"mirna", "gene", "label"})

    _, stage2_x, _, stage2_cols = build_demo_feature_table()
    stage2_x = stage2_x[stage2_cols].astype(np.float64)

    summary = {
        "stage1": audit_matrix("stage1_train", stage1_x),
        "stage2": audit_matrix("stage2_demo", stage2_x),
        "notes": [
            "Audit uses training data for stage 1 to avoid validation/test leakage.",
            "Near-constant is defined as a most-common value frequency >= 99.5%.",
            "High correlation is absolute Pearson correlation >= 0.95.",
            "High correlation does not always mean the feature is invalid, but it indicates possible redundancy.",
        ],
    }
    (OUT_DIR / "feature_redundancy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
