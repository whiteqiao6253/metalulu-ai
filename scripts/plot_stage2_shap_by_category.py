from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


IMPORTANCE_PATH = Path(
    r"D:\nn\process\shap_results\stage2_tabpfn_realthermo_pruned"
    r"\stage2_realthermo_pruned_tabpfn_shap_importance.csv"
)
OUT_DIR = Path(r"D:\nn\process\shap_results\stage2_tabpfn_realthermo_pruned")
PAPER_FIGURE_PATH = Path(r"D:\nn\figures\figure_stage2_shap_importance.png")


CATEGORY_COLORS = {
    "miRNA intrinsic": "#4C78A8",
    "UTR context": "#59A14F",
    "Seed/site": "#F28E2B",
    "Local context": "#76B7B2",
    "Thermodynamics": "#8E63B6",
    "Stage 1 score": "#E15759",
    "Transcript statistic": "#7F7F7F",
    "Supplementary pairing": "#9C755F",
}


def feature_category(feature: str) -> str:
    thermo_tokens = ("MFE", "p_value", "delta_mfe", "duplex")
    if feature == "best_score":
        return "Stage 1 score"
    if feature == "transcript_count":
        return "Transcript statistic"
    if any(token in feature for token in thermo_tokens) or feature == "seed_duplex_proxy":
        return "Thermodynamics"
    if feature.startswith("mirna_"):
        return "miRNA intrinsic"
    if feature.startswith("utr_"):
        return "UTR context"
    if feature.startswith(("local_", "upstream_", "downstream_")):
        return "Local context"
    if (
        feature.startswith("site_")
        or feature.startswith("seed_")
        or feature.startswith("num_")
        or feature in {"has_high_quality_site", "best_site_weight", "best_partial_seed_matches"}
    ):
        return "Seed/site"
    if feature == "supplementary_pairing_13_16":
        return "Supplementary pairing"
    return "Seed/site"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    importance = pd.read_csv(IMPORTANCE_PATH).head(30).copy()
    importance["category"] = importance["feature"].map(feature_category)
    plot_df = importance.iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(10.2, 9.4))
    colors = plot_df["category"].map(CATEGORY_COLORS)
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    ax.set_title("Stage 2 TabPFN SHAP Importance by Feature Category")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for y, value in enumerate(plot_df["mean_abs_shap"]):
        ax.text(value + 0.001, y, f"{value:.3f}", va="center", fontsize=8)

    used_categories = [cat for cat in CATEGORY_COLORS if cat in set(importance["category"])]
    handles = [mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat) for cat in used_categories]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

    fig.tight_layout()
    out_path = OUT_DIR / "stage2_realthermo_tabpfn_shap_bar_by_category.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    fig.savefig(PAPER_FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"saved={out_path}")
    print(f"saved={PAPER_FIGURE_PATH}")


if __name__ == "__main__":
    main()
