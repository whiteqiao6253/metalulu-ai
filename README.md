# MetaLulu-AI Supplementary Package

This folder contains the final supplemental materials aligned with the manuscript.

## Figures
- `fig1_framework.png`
- `fig2_shap_stage1.png`
- `fig3_shap_stage2.png`
- `fig4_split_comparison.png`
- `fig5_stage2_gain.png`
- `fig6_gain_vs_shap.png`
- `fig7_active_learning_loop.jpeg`
- `fig8_baseline_comparison.png`
- `fig8_benchmark_comparison.png`
- `fig9_calibration.png`
- `figure_stage2_adaptation_performance.png`
- `figure_stage2_shap_importance.png`
- `stage1_final_performance.png`

## Tables / Summaries
- `baseline_comparison_metrics.csv`
- `stage1_full_metrics.csv`
- `stage2_logo_results.csv`
- `stage2_loso_results.csv`
- `reviewer_supplement_summary.json`

## Reproducible Scripts
- `scripts/build_bio_features.py`
- `scripts/train_xgboost_bio.py`
- `scripts/train_tabpfn_demo_realthermo_pruned.py`
- `scripts/score_sequence_pair_two_stage.py`
- `scripts/make_group_splits.py`
- `scripts/audit_label_leakage.py`
- `scripts/audit_feature_redundancy.py`
- `scripts/cv_tabpfn_demo_eval.py`
- `scripts/repeat_tabpfn_demo_eval.py`
- `scripts/plot_stage2_shap_by_category.py`

All files are the final MetaLulu-AI versions intended to match the manuscript text and figure references.

## Reproducibility Note
The scripts compile successfully, but the full end-to-end pipeline still depends on the same training/intermediate data layout used in the working manuscript environment. For a public GitHub release, the code should be treated as the executable analysis layer, while large raw datasets are handled separately.
