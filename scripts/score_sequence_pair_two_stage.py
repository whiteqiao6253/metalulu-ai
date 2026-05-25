import argparse
import inspect
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.utils.validation as sk_validation
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from build_bio_features import build_feature_record, normalize_rna


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

from train_tabpfn_demo import build_demo_feature_table


ROOT = Path(r"D:\nn\process")
XGB_DIR = ROOT / "xgb_results"
DATA_DIR = Path(r"D:\nn\data")


@lru_cache(maxsize=1)
def load_stage1_model_and_threshold():
    metrics = json.loads((XGB_DIR / "metrics.json").read_text(encoding="utf-8"))
    model = XGBClassifier(**metrics["chosen_params"])
    model.load_model(XGB_DIR / "best_model.json")

    recommended_path = XGB_DIR / "recommended_metrics.csv"
    if recommended_path.exists():
        rec = pd.read_csv(recommended_path)
        test_row = rec[rec["split"] == "test"]
        if not test_row.empty:
            threshold = float(test_row.iloc[0]["threshold"])
        else:
            threshold = float(metrics["chosen_threshold"])
    else:
        threshold = float(metrics["chosen_threshold"])
    return model, threshold


def build_stage1_feature_row(mirna_seq: str, target_seq: str, gene_name: str):
    cleaned_mirna = normalize_rna(mirna_seq)
    cleaned_target = normalize_rna(target_seq)
    if not cleaned_mirna:
        raise ValueError("miRNA sequence is empty after normalization.")
    if not cleaned_target:
        raise ValueError("Target sequence is empty after normalization.")

    feature_row = build_feature_record(
        {
            "mirna": "custom_mirna",
            "gene": gene_name,
            "label": 0,
            "mirna_seq": cleaned_mirna,
            "utr_seq": cleaned_target,
        }
    )
    feature_df = pd.DataFrame([feature_row])
    x_stage1 = feature_df.drop(columns=["mirna", "gene", "label"]).astype(np.float32)
    return feature_df, x_stage1, cleaned_mirna, cleaned_target


def apply_legacy_demo_feature_convention(feature_df: pd.DataFrame) -> pd.DataFrame:
    legacy_df = feature_df.copy()
    legacy_defaults = {
        "mirna_self_MFE_per_nt": 0.0,
        "utr_head_MFE_per_nt": 0.0,
        "duplex_MFE_per_nt": 0.0,
        "duplex_p_value": 1.0,
        "delta_mfe_norm": 0.0,
        "seed_duplex_proxy": 0.0,
    }
    for col, value in legacy_defaults.items():
        if col in legacy_df.columns:
            legacy_df[col] = value
    return legacy_df


@lru_cache(maxsize=1)
def load_stage2_training():
    demo_df, x_demo, y_demo, feature_cols = build_demo_feature_table()
    model = TabPFNClassifier(device="cuda", N_ensemble_configurations=16, seed=42)
    model.fit(x_demo, y_demo)
    return demo_df, x_demo, y_demo, feature_cols, model


@lru_cache(maxsize=1)
def load_stage2_demo_fixed_split():
    demo_df, x_demo, y_demo, feature_cols = build_demo_feature_table()
    indices = np.arange(len(demo_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=y_demo,
    )
    model = TabPFNClassifier(device="cuda", N_ensemble_configurations=16, seed=42)
    model.fit(x_demo.iloc[train_idx], y_demo[train_idx])
    return demo_df.reset_index(drop=True), x_demo.reset_index(drop=True), y_demo, feature_cols, model, set(test_idx.tolist())


def build_stage2_feature_row(stage1_feature_df: pd.DataFrame, stage1_score: float, gene_name: str, feature_cols):
    base = stage1_feature_df.iloc[0].to_dict()

    stage2_row = {
        "transcript_count": 1.0,
        "max_score": stage1_score,
        "mean_score": stage1_score,
        "min_score": stage1_score,
        "best_score": stage1_score,
        "score": stage1_score,
    }

    for key, value in base.items():
        if key not in {"mirna", "gene", "label"}:
            stage2_row[key] = value

    for col in feature_cols:
        if col.startswith("gene_"):
            stage2_row[col] = 1.0 if col == f"gene_{gene_name}" else 0.0

    for col in feature_cols:
        stage2_row.setdefault(col, 0.0)

    row_df = pd.DataFrame([[stage2_row[col] for col in feature_cols]], columns=feature_cols).astype(np.float32)
    return row_df


@lru_cache(maxsize=1)
def load_demo_frozen_tables():
    transcript_scores = pd.read_csv(XGB_DIR / "demo_external_transcript_scores.csv")
    summary = pd.read_csv(XGB_DIR / "demo_external_summary.csv")
    target_df = pd.read_csv(DATA_DIR / "target.csv")
    target_df["gene_key"] = target_df["target"].astype(str).str.upper()
    target_df["utr_key"] = target_df["utrSeq"].astype(str).map(normalize_rna)
    return transcript_scores, summary, target_df


def resolve_transcript_id(gene_name: str, target_seq: str, target_df: pd.DataFrame):
    gene_key = str(gene_name).upper()
    utr_key = normalize_rna(target_seq)
    match = target_df[(target_df["gene_key"] == gene_key) & (target_df["utr_key"] == utr_key)]
    if match.empty:
        return None
    return str(match.iloc[0]["transcriptId"])


def lookup_demo_compatible_scores(mirna_seq: str, target_seq: str, gene_name: str):
    transcript_scores, summary, target_df = load_demo_frozen_tables()
    transcript_id = resolve_transcript_id(gene_name, target_seq, target_df)
    if transcript_id is None:
        return None

    transcript_match = transcript_scores[
        (transcript_scores["gene_name"] == gene_name)
        & (transcript_scores["mirna_seq"] == mirna_seq)
        & (transcript_scores["transcriptId"].astype(str) == transcript_id)
    ].copy()
    if transcript_match.empty:
        return None

    transcript_row = transcript_match.iloc[0]
    query_id = int(transcript_row["query_id"])

    demo_df, x_demo, _, _, model, test_idx = load_stage2_demo_fixed_split()
    demo_row = demo_df[demo_df["query_id"] == query_id]
    if demo_row.empty:
        return None
    demo_row = demo_row.iloc[0]
    stage2_score = float(model.predict_proba(x_demo.iloc[[int(demo_row.name)]])[:, 1][0])

    stage1_feature_snapshot = transcript_row.drop(
        labels=[
            "query_id",
            "gene_name",
            "matched_gene",
            "transcriptId",
            "mirna_seq",
            "input_label",
            "score",
            "pred_label",
            "threshold_used",
        ],
        errors="ignore",
    ).to_dict()

    return {
        "query_id": query_id,
        "transcript_id": transcript_id,
        "stage1_score": float(transcript_row["score"]),
        "stage1_pred_label": int(transcript_row["pred_label"]),
        "stage1_threshold": float(transcript_row["threshold_used"]),
        "stage2_score": stage2_score,
        "stage2_pred_label": int(stage2_score >= 0.5),
        "stage2_threshold": 0.5,
        "final_score": stage2_score,
        "final_pred_label": int(stage2_score >= 0.5),
        "stage1_feature_snapshot": stage1_feature_snapshot,
        "demo_fixed_split_match": query_id in test_idx,
    }


def score_two_stage_pair(mirna_seq: str, target_seq: str, gene_name: str = "custom_target", mode: str = "legacy_demo_compatible"):
    cleaned_mirna = normalize_rna(mirna_seq)
    cleaned_target = normalize_rna(target_seq)

    if mode in {"demo_replay", "demo_replay_if_available"}:
        frozen = lookup_demo_compatible_scores(cleaned_mirna, cleaned_target, gene_name)
        if frozen is not None:
            return {
                "gene_name": gene_name,
                "mirna_seq_used": cleaned_mirna,
                "target_seq_used": cleaned_target,
                "mode_used": "demo_replay",
                **frozen,
            }
        if mode == "demo_replay":
            raise ValueError("Input pair was not found in the frozen demo replay artifacts.")

    stage1_model, stage1_threshold = load_stage1_model_and_threshold()
    stage1_feature_df, x_stage1, cleaned_mirna, cleaned_target = build_stage1_feature_row(
        mirna_seq, target_seq, gene_name
    )
    if mode == "legacy_demo_compatible":
        stage1_feature_df = apply_legacy_demo_feature_convention(stage1_feature_df)
        x_stage1 = stage1_feature_df.drop(columns=["mirna", "gene", "label"]).astype(np.float32)
    stage1_score = float(stage1_model.predict_proba(x_stage1)[:, 1][0])
    stage1_pred = int(stage1_score >= stage1_threshold)

    if mode == "legacy_demo_compatible":
        _, _, _, stage2_feature_cols, stage2_model, _ = load_stage2_demo_fixed_split()
    else:
        _, _, _, stage2_feature_cols, stage2_model = load_stage2_training()
    x_stage2 = build_stage2_feature_row(stage1_feature_df, stage1_score, gene_name, stage2_feature_cols)
    stage2_score = float(stage2_model.predict_proba(x_stage2)[:, 1][0])
    stage2_pred = int(stage2_score >= 0.5)

    result = {
        "gene_name": gene_name,
        "mirna_seq_used": cleaned_mirna,
        "target_seq_used": cleaned_target,
        "mode_used": mode,
        "stage1_score": stage1_score,
        "stage1_pred_label": stage1_pred,
        "stage1_threshold": stage1_threshold,
        "stage2_score": stage2_score,
        "stage2_pred_label": stage2_pred,
        "stage2_threshold": 0.5,
        "final_score": stage2_score,
        "final_pred_label": stage2_pred,
        "stage1_feature_snapshot": stage1_feature_df.iloc[0].to_dict(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage scorer: stage 1 XGBoost bioinformatics model + stage 2 TabPFN experimental adapter."
    )
    parser.add_argument("--mirna-seq", default=None, help="miRNA sequence")
    parser.add_argument("--target-seq", default=None, help="Target sequence, preferably 3'UTR")
    parser.add_argument("--gene-name", default="custom_target", help="Gene name for display and stage-2 gene encoding")
    parser.add_argument(
        "--mode",
        default="legacy_demo_compatible",
        choices=["demo_replay_if_available", "demo_replay", "production_all_demo", "legacy_demo_compatible"],
        help="Use frozen demo replay artifacts when available, or choose production/all-demo or legacy demo-compatible feature conventions.",
    )
    parser.add_argument("--show-features", action="store_true", help="Print selected stage-1 features")
    args = parser.parse_args()

    mirna_seq = args.mirna_seq or input("miRNA sequence: ").strip()
    target_seq = args.target_seq or input("Target sequence (3'UTR preferred): ").strip()

    result = score_two_stage_pair(mirna_seq, target_seq, gene_name=args.gene_name, mode=args.mode)

    print(f"gene_name: {result['gene_name']}")
    print(f"mode_used: {result['mode_used']}")
    print(f"stage1_score: {result['stage1_score']:.6f}")
    print(f"stage1_pred_label: {result['stage1_pred_label']}")
    print(f"stage1_threshold: {result['stage1_threshold']:.6f}")
    print(f"stage2_score: {result['stage2_score']:.6f}")
    print(f"stage2_pred_label: {result['stage2_pred_label']}")
    print(f"stage2_threshold: {result['stage2_threshold']:.6f}")
    print(f"final_score: {result['final_score']:.6f}")
    print(f"final_pred_label: {result['final_pred_label']}")
    print(f"mirna_seq_used: {result['mirna_seq_used']}")
    print(f"target_seq_used_length: {len(result['target_seq_used'])}")

    if args.show_features:
        snapshot = result["stage1_feature_snapshot"]
        keys = [
            "num_seed_matches",
            "num_8mer_sites",
            "num_7mer_m8_sites",
            "num_7mer_A1_sites",
            "has_high_quality_site",
            "best_site_weight",
            "duplex_MFE_per_nt",
            "seed_duplex_proxy",
            "local_AU_content",
            "site_position_norm",
        ]
        print("stage1_key_features:")
        for key in keys:
            if key in snapshot:
                print(f"  {key}: {snapshot[key]}")


if __name__ == "__main__":
    main()
