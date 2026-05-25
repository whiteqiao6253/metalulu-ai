import argparse
import math
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd


COMPLEMENT = str.maketrans("AUGC", "UACG")
SITE_TYPE_WEIGHTS = {
    "8mer": 1.0,
    "7mer_m8": 0.8,
    "7mer_A1": 0.6,
    "6mer": 0.4,
    "none": 0.0,
}
RNAHYBRID_MAX_UTR = 3000
RNAFOLD_MAX_UTR = 400
LOCAL_FLANK = 30
CONTEXT_FLANK = 70
DUPLEX_WINDOW_FLANK = 20


def normalize_rna(seq: str) -> str:
    seq = str(seq).upper().replace("T", "U")
    return "".join(base for base in seq if base in "ACGU")


def reverse_complement_rna(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def au_content(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("A") + seq.count("U")) / len(seq)


def dinuc_freq(seq: str, motifs) -> float:
    if len(seq) < 2:
        return 0.0
    hits = 0
    for i in range(len(seq) - 1):
        if seq[i : i + 2] in motifs:
            hits += 1
    return hits / (len(seq) - 1)


def bounded_log10(value: int, scale: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log10(value) / scale, 1.0)


def slice_safe(seq: str, start: int, end: int) -> str:
    return seq[max(0, start) : max(0, end)]


def canonical_sites(mirna_seq: str, utr_seq: str):
    sites = []
    if len(mirna_seq) < 8 or len(utr_seq) < 6:
        return sites

    seed_2_8 = reverse_complement_rna(mirna_seq[1:8])
    seed_2_7 = reverse_complement_rna(mirna_seq[1:7])
    seen = set()

    start = 0
    while True:
        pos = utr_seq.find(seed_2_8, start)
        if pos == -1:
            break
        site_type = "8mer" if pos + 7 < len(utr_seq) and utr_seq[pos + 7] == "A" else "7mer_m8"
        sites.append({"start": pos, "length": 8 if site_type == "8mer" else 7, "type": site_type})
        seen.add((pos, 7))
        start = pos + 1

    start = 0
    while True:
        pos = utr_seq.find(seed_2_7, start)
        if pos == -1:
            break
        if (pos, 7) not in seen:
            site_type = "7mer_A1" if pos + 6 < len(utr_seq) and utr_seq[pos + 6] == "A" else "6mer"
            sites.append({"start": pos, "length": 7 if site_type == "7mer_A1" else 6, "type": site_type})
        start = pos + 1

    return sorted(sites, key=lambda x: (x["start"], -SITE_TYPE_WEIGHTS[x["type"]]))


def seed_match_pairs(mirna_seq: str, target_window: str) -> int:
    seed = mirna_seq[1:8]
    target_rc = reverse_complement_rna(target_window[:7])
    total = 0
    for left, right in zip(seed, target_rc):
        if left == right:
            total += 1
    return total


def fallback_site(mirna_seq: str, utr_seq: str):
    if len(utr_seq) < 6 or len(mirna_seq) < 8:
        return {"start": 0, "length": 0, "type": "none", "pairs": 0}

    best = {"start": 0, "length": 6, "type": "none", "pairs": -1}
    for win_len in (8, 7, 6):
        if len(utr_seq) < win_len:
            continue
        for pos in range(len(utr_seq) - win_len + 1):
            window = utr_seq[pos : pos + win_len]
            pairs = seed_match_pairs(mirna_seq, window)
            if pairs > best["pairs"]:
                best = {"start": pos, "length": win_len, "type": "none", "pairs": pairs}
    return best


def choose_best_site(mirna_seq: str, utr_seq: str, sites):
    if not sites:
        return fallback_site(mirna_seq, utr_seq)

    ranked = []
    for site in sites:
        window = utr_seq[site["start"] : site["start"] + site["length"]]
        flank = slice_safe(utr_seq, site["start"] - LOCAL_FLANK, site["start"] + site["length"] + LOCAL_FLANK)
        ranked.append(
            (
                SITE_TYPE_WEIGHTS[site["type"]],
                au_content(flank),
                -site["start"],
                {**site, "pairs": seed_match_pairs(mirna_seq, window)},
            )
        )
    ranked.sort(reverse=True)
    return ranked[0][3]


def supplementary_pairing(mirna_seq: str, utr_seq: str, site_start: int, site_length: int) -> int:
    supp = mirna_seq[12:16]
    if not supp:
        return 0
    target = slice_safe(utr_seq, site_start + site_length, site_start + site_length + len(supp))
    target_rc = reverse_complement_rna(target)
    total = 0
    for left, right in zip(supp, target_rc):
        if left == right:
            total += 1
    return total


def run_command(cmd, input_text=None, timeout=20):
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="ignore",
    )
    return result.stdout


def run_wsl_bash(script: str, timeout=20):
    result = subprocess.run(
        ["wsl.exe", "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="ignore",
    )
    return result.stdout


@lru_cache(maxsize=None)
def rnafold_stability(seq: str) -> float:
    seq = normalize_rna(seq)
    if not seq:
        return 0.0
    script = f"printf '%s\\n' {shlex.quote(seq)} | RNAfold --noPS"
    output = run_wsl_bash(script, timeout=20)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\)", lines[1])
    if not match:
        return 0.0
    mfe = float(match.group(1))
    return max(-mfe / max(len(seq), 1), 0.0)


@lru_cache(maxsize=None)
def rnahybrid_metrics(mirna_seq: str, utr_seq: str):
    mirna_seq = normalize_rna(mirna_seq)
    utr_seq = normalize_rna(utr_seq)[:RNAHYBRID_MAX_UTR]
    if not mirna_seq or not utr_seq:
        return 0.0, 1.0
    script = "RNAhybrid -s 3utr_human -c {utr} {mirna}".format(
        utr=shlex.quote(utr_seq),
        mirna=shlex.quote(mirna_seq),
    )
    output = run_wsl_bash(script, timeout=20)
    line = output.strip().splitlines()[0] if output.strip() else ""
    parts = line.split(":")
    if len(parts) < 6:
        return 0.0, 1.0
    try:
        mfe = float(parts[4])
        p_value = float(parts[5])
    except ValueError:
        return 0.0, 1.0
    return max(-mfe / max(len(mirna_seq), 1), 0.0), p_value


def build_feature_record(row):
    mirna_seq = normalize_rna(row["mirna_seq"])
    utr_seq = normalize_rna(row["utr_seq"])
    sites = canonical_sites(mirna_seq, utr_seq)
    best_site = choose_best_site(mirna_seq, utr_seq, sites)

    site_start = int(best_site["start"])
    site_length = int(best_site["length"])
    site_end = site_start + site_length
    site_seq = utr_seq[site_start:site_end]

    local_seq = slice_safe(utr_seq, site_start - LOCAL_FLANK, site_end + LOCAL_FLANK)
    upstream_seq = slice_safe(utr_seq, site_start - CONTEXT_FLANK, site_start)
    downstream_seq = slice_safe(utr_seq, site_end, site_end + CONTEXT_FLANK)
    seed_window = utr_seq[site_start : site_start + min(7, max(site_length, 7))]
    duplex_window = slice_safe(utr_seq, site_start - DUPLEX_WINDOW_FLANK, site_end + DUPLEX_WINDOW_FLANK)

    mirna_self = rnafold_stability(mirna_seq)
    utr_head = rnafold_stability(utr_seq[:RNAFOLD_MAX_UTR])
    duplex_mfe, duplex_p = rnahybrid_metrics(mirna_seq, duplex_window if duplex_window else utr_seq[:40])
    seed_duplex_proxy = duplex_mfe * (best_site.get("pairs", 0) / 7.0)

    site_type = best_site["type"]
    total_sites = len(sites)
    num_8mer = sum(1 for site in sites if site["type"] == "8mer")
    num_7mer_m8 = sum(1 for site in sites if site["type"] == "7mer_m8")
    num_7mer_A1 = sum(1 for site in sites if site["type"] == "7mer_A1")
    num_6mer = sum(1 for site in sites if site["type"] == "6mer")

    record = {
        "mirna": row["mirna"],
        "gene": row["gene"],
        "label": int(row["label"]),
        "mirna_GC_content": gc_content(mirna_seq),
        "mirna_AU_content": au_content(mirna_seq),
        "mirna_pos1_U": 1.0 if mirna_seq[:1] == "U" else 0.0,
        "mirna_pos1_A": 1.0 if mirna_seq[:1] == "A" else 0.0,
        "mirna_pos2_G": 1.0 if mirna_seq[1:2] == "G" else 0.0,
        "mirna_seed_GC": gc_content(mirna_seq[1:8]),
        "mirna_seed_AU": au_content(mirna_seq[1:8]),
        "mirna_3p_GC": gc_content(mirna_seq[-7:]),
        "mirna_length_norm": min(len(mirna_seq) / 25.0, 1.0),
        "mirna_self_MFE_per_nt": mirna_self,
        "utr_GC_content": gc_content(utr_seq),
        "utr_AU_content": au_content(utr_seq),
        "utr_length_raw": float(len(utr_seq)),
        "utr_length_log10_norm": bounded_log10(len(utr_seq), 4.0),
        "utr_CG_dinuc_freq": dinuc_freq(utr_seq, {"CG"}),
        "utr_AU_dinuc_freq": dinuc_freq(utr_seq, {"AU", "UA"}),
        "utr_head_MFE_per_nt": utr_head,
        "num_seed_matches": float(total_sites),
        "num_8mer_sites": float(num_8mer),
        "num_7mer_m8_sites": float(num_7mer_m8),
        "num_7mer_A1_sites": float(num_7mer_A1),
        "num_6mer_sites": float(num_6mer),
        "has_high_quality_site": 1.0 if (num_8mer + num_7mer_m8 + num_7mer_A1) > 0 else 0.0,
        "best_site_weight": SITE_TYPE_WEIGHTS[site_type],
        "num_sites_log": math.log1p(total_sites),
        "best_partial_seed_matches": float(best_site.get("pairs", 0)),
        "site_type_weight": SITE_TYPE_WEIGHTS[site_type],
        "site_8mer": 1.0 if site_type == "8mer" else 0.0,
        "site_7mer_m8": 1.0 if site_type == "7mer_m8" else 0.0,
        "site_7mer_A1": 1.0 if site_type == "7mer_A1" else 0.0,
        "site_6mer": 1.0 if site_type == "6mer" else 0.0,
        "site_start": float(site_start),
        "site_end": float(site_end),
        "site_t1_is_A": 1.0 if site_end > site_start and site_end <= len(utr_seq) and utr_seq[site_end - 1 : site_end] == "A" else 0.0,
        "site_position_norm": (site_start + 0.5 * max(site_length, 1)) / max(len(utr_seq), 1),
        "site_dist_from_end_norm": (len(utr_seq) - site_end) / max(len(utr_seq), 1),
        "local_AU_content": au_content(local_seq),
        "local_GC_content": gc_content(local_seq),
        "upstream_AU_content": au_content(upstream_seq),
        "downstream_AU_content": au_content(downstream_seq),
        "upstream_GC_content": gc_content(upstream_seq),
        "downstream_GC_content": gc_content(downstream_seq),
        "site_GC_content": gc_content(site_seq),
        "seed_GC_content": gc_content(seed_window),
        "seed_AU_content": au_content(seed_window),
        "supplementary_pairing_13_16": float(supplementary_pairing(mirna_seq, utr_seq, site_start, site_length)),
        "site_length_norm": site_length / 8.0,
        "duplex_MFE_per_nt": duplex_mfe,
        "duplex_p_value": duplex_p,
        "delta_mfe_norm": duplex_mfe - mirna_self,
        "seed_duplex_proxy": seed_duplex_proxy,
    }
    return record


def sample_frame(df: pd.DataFrame, per_label: int, random_state: int):
    pieces = []
    for label in sorted(df["label"].unique()):
        part = df[df["label"] == label]
        take = min(per_label, len(part))
        pieces.append(part.sample(n=take, random_state=random_state))
    return pd.concat(pieces, ignore_index=True)


def process_frame(df: pd.DataFrame, workers: int):
    rows = [row for _, row in df.iterrows()]
    start = time.time()
    if workers <= 1:
        records = [build_feature_record(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(build_feature_record, rows))
    out = pd.DataFrame(records)
    elapsed = time.time() - start
    return out, elapsed


def diagnose_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if column in {"mirna", "gene"}:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
                "null_pct": series.isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=r"D:\nn\process\train.csv")
    parser.add_argument("--val", default=r"D:\nn\process\val.csv")
    parser.add_argument("--test", default=r"D:\nn\process\test.csv")
    parser.add_argument("--out-dir", default=r"D:\nn\process\bio_features")
    parser.add_argument("--sample-per-label", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--only-splits", default="train,val,test")
    args = parser.parse_args()

    datasets = {
        "train": pd.read_csv(args.train),
        "val": pd.read_csv(args.val),
        "test": pd.read_csv(args.test),
    }
    requested_splits = {item.strip() for item in str(args.only_splits).split(",") if item.strip()}

    Path = __import__("pathlib").Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for split_name, df in datasets.items():
        if split_name not in requested_splits:
            continue
        work_df = sample_frame(df, args.sample_per_label, args.random_state) if args.sample_per_label > 0 else df
        features_df, elapsed = process_frame(work_df, workers=args.workers)
        out_path = out_dir / f"{split_name}_features.csv"
        diag_path = out_dir / f"{split_name}_diagnosis.csv"
        features_df.to_csv(out_path, index=False)
        diagnose_features(features_df).to_csv(diag_path, index=False)
        manifest.append(
            {
                "split": split_name,
                "rows": len(work_df),
                "seconds": round(elapsed, 2),
                "output": str(out_path),
                "diagnosis": str(diag_path),
            }
        )

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
