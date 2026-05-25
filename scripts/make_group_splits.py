import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FILES = [
    Path(r"D:\nn\process\bio_features_full\train_features.csv"),
    Path(r"D:\nn\process\bio_features_full\val_features.csv"),
    Path(r"D:\nn\process\bio_features_full\test_features.csv"),
]


def load_all_features():
    frames = [pd.read_csv(path) for path in SOURCE_FILES]
    return pd.concat(frames, ignore_index=True)


def assign_groups(unique_groups, seed, train_frac=0.7, val_frac=0.1):
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_groups, dtype=object)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train : n_train + n_val])
    test_groups = set(shuffled[n_train + n_val :])
    return train_groups, val_groups, test_groups


def split_by_group(df, group_col, seed):
    unique_groups = df[group_col].dropna().astype(str).unique().tolist()
    train_groups, val_groups, test_groups = assign_groups(unique_groups, seed=seed)

    working = df.copy()
    group_values = working[group_col].astype(str)
    split = np.where(
        group_values.isin(train_groups),
        "train",
        np.where(group_values.isin(val_groups), "val", "test"),
    )
    working["split"] = split
    return working


def summarize(df, group_col):
    rows = []
    for split_name in ["train", "val", "test"]:
        part = df[df["split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "rows": len(part),
                "positives": int(part["label"].sum()),
                "negatives": int((1 - part["label"]).sum()),
                "unique_groups": part[group_col].astype(str).nunique(),
                "unique_genes": part["gene"].astype(str).nunique(),
                "unique_mirnas": part["mirna"].astype(str).nunique(),
            }
        )
    return pd.DataFrame(rows)


def check_overlap(df, group_col):
    groups = {
        split: set(df.loc[df["split"] == split, group_col].astype(str))
        for split in ["train", "val", "test"]
    }
    overlaps = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlaps.append({"pair": f"{a}-{b}", "group_overlap": len(groups[a] & groups[b])})
    return pd.DataFrame(overlaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-col", choices=["gene", "mirna"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_features()
    grouped = split_by_group(df, group_col=args.group_col, seed=args.seed)

    summary = summarize(grouped, args.group_col)
    overlap = check_overlap(grouped, args.group_col)

    for split_name in ["train", "val", "test"]:
        part = grouped[grouped["split"] == split_name].drop(columns=["split"])
        part.to_csv(out_dir / f"{split_name}_features.csv", index=False)

    summary.to_csv(out_dir / "split_summary.csv", index=False)
    overlap.to_csv(out_dir / "overlap_check.csv", index=False)

    print(summary.to_string(index=False))
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
