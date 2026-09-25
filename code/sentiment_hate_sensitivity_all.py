"""All sensitivity checks for the latest sentiment/hate-speech dyadic DiD.

One standalone code file.  It preserves the two previously verified analysis
implementations below, while forcing both to read the September 2026 source
workbooks used by the latest main model.  It never edits those source files.

Checks: post-count weighting; dyad, origin, mentioned and two-way country
clustering; raw-rate functional form; placebo periods; +20 minimum posts;
exclusion of the two directed Russia/Ukraine dyads; leave one country out.

Usage: python sentiment_hate_sensitivity_all.py [--mode all|checks|dyads]
       [--output-dir DIRECTORY] [--figures]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _load_embedded(name: str):
    """Compile a verbatim, readable source block in its own module namespace."""
    import linecache
    import types

    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    start = lines.index(f"# === BEGIN EMBEDDED {name} ===") + 1
    stop = lines.index(f"# === END EMBEDDED {name} ===")
    source = "\n".join(line[2:] for line in lines[start:stop]) + "\n"
    virtual_name = f"<embedded {name}>"
    linecache.cache[virtual_name] = (
        len(source), None, source.splitlines(keepends=True), virtual_name
    )
    module = types.ModuleType(name)
    module.__file__ = str(Path(__file__).resolve())
    sys.modules[name] = module
    exec(compile(source, virtual_name, "exec"), module.__dict__)
    return module


def _validate_inputs():
    paths = [
        DATA_DIR / "pair_results_by_period.xlsx",
        DATA_DIR / "pair_results_by_year.xlsx",
        DATA_DIR / "asylum_data.xlsx",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing current DID source file(s): " + "; ".join(missing))
    asylum = pd.read_excel(paths[2], sheet_name="2022")
    required = ["first_time_applicants_total", "non-Ukrainian_first_time_applicant", "first_time_applicants_ukrainian"]
    if any(column not in asylum.columns for column in required):
        raise ValueError("2022 asylum sheet lacks expected treatment columns")
    difference = (
        asylum["first_time_applicants_total"] - asylum["non-Ukrainian_first_time_applicant"]
        - asylum["first_time_applicants_ukrainian"]
    )
    if difference.isna().any() or (difference.abs() > 1e-9).any():
        raise ValueError(
            "Expected 2022 first_time_applicants_total = non-Ukrainian non-Ukrainian_first_time_applicant + "
            "first_time_applicants_ukrainian in every country"
        )
    return paths


def _combine_results(root: Path) -> Path:
    sheets = {}
    workbook = root / "checks" / "robustness_significance_summary_concise.xlsx"
    if workbook.is_file():
        sheets["checks"] = pd.read_excel(workbook, sheet_name="concise_side_summary")
    geo = root / "dyads" / "geopolitical_sensitivity"
    for name, filename in [
        ("dyad_exclusion", "sensitivity_post_coefficients_long.csv"),
        ("leave_one_out", "leave_one_country_out_influence_summary.csv"),
        ("sample_audit", "sensitivity_sample_audit.csv"),
    ]:
        path = geo / filename
        if path.is_file():
            sheets[name] = pd.read_csv(path)
    if "dyad_exclusion" in sheets:
        all_rows = sheets["dyad_exclusion"]
        wide_frames = []
        for year in (2015, 2022):
            wide_path = geo / f"sensitivity_results_{year}crisis_wide.csv"
            if wide_path.is_file():
                wide_frames.append(pd.read_csv(wide_path))
        wide = pd.concat(wide_frames, ignore_index=True) if wide_frames else None
        focal_rows = []
        for crisis, treatment in [
            (2015, "first_time_applicants"),
            (2022, "non-Ukrainian_first_time_applicant"),
            (2022, "first_time_applicants_ukrainian"),
        ]:
            relevant = all_rows.loc[
                (all_rows["crisis"] == crisis)
                & (all_rows["treatment"] == treatment)
                & (all_rows["outcome"] == "mean_sentiment_score")
                & (all_rows["side"] == "mentioned")
            ]
            full = relevant.loc[relevant["sample_spec"] == "full_sample"].iloc[0]
            s1 = relevant.loc[
                relevant["sample_spec"] == "S1_exclude_direct_Russia_Ukraine_dyads"
            ].iloc[0]
            loo = relevant.loc[
                relevant["sample_spec"] == "S3_leave_one_country_out"
            ]
            loo_final_count = None
            if wide is not None:
                loo_wide = wide.loc[
                    (wide["crisis"] == crisis)
                    & (wide["treatment"] == treatment)
                    & (wide["outcome"] == "mean_sentiment_score")
                    & (wide["sample_spec"] == "S3_leave_one_country_out")
                ]
                post_p_columns = [
                    column for column in loo_wide.columns
                    if column.startswith("mentioned_post_p_")
                ]
                post_p = loo_wide[post_p_columns].bfill(axis=1).iloc[:, 0]
                loo_final_count = int((
                    (post_p < 0.05)
                    & (loo_wide["mentioned_parallel"] == "pass")
                ).sum())
            focal_rows.append({
                "crisis": crisis,
                "treatment": treatment,
                "outcome": "external sentiment",
                "unit": "scale points on -1 to 1",
                "main_beta": full["coef"],
                "main_p": full["p"],
                "S1_beta": s1["coef"],
                "S1_p": s1["p"],
                "LOO_models": len(loo),
                "LOO_same_sign": int(loo["same_sign_as_full"].fillna(False).sum()),
                "LOO_CI_excludes_zero": int(loo["ci_excludes_zero"].fillna(False).sum()),
                "LOO_post_significant_and_pretrend_nonsignificant": loo_final_count,
            })
        sheets["focal_results"] = pd.DataFrame(focal_rows)
    if not sheets:
        raise RuntimeError("No sensitivity results were produced")
    output = root / "sentiment_hate_sensitivity_combined.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "checks", "dyads"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    root = args.output_dir or Path(__file__).resolve().parent.parent / "results" / (
        "sentiment_hate_sensitivity_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=True)
    paths = _validate_inputs()

    # The original geopolitical module creates OUT_ROOT during import.
    os.environ["DID_OUT_ROOT"] = str(root / "dyads")
    checks = _load_embedded("sentiment_checks")
    dyads = _load_embedded("sentiment_dyads")
    for module in (checks, dyads):
        module.combined_path = str(paths[0])
        module.yearly_path = str(paths[1])
        module.asylum_path = str(paths[2])
    checks.OUT_ROOT = str(root / "checks")
    dyads.OUT_ROOT = str(root / "dyads")
    if not args.figures:
        # Suppress only optional plots; estimation and tabular exports are unchanged.
        checks.draw_all_figures = lambda *unused_args, **unused_kwargs: None
        dyads.draw_all_geopolitical_sensitivity_figures = (
            lambda *unused_args, **unused_kwargs: None
        )

    if args.mode in ("all", "checks"):
        checks.main()
    if args.mode in ("all", "dyads"):
        dyads.run_all_geopolitical_sensitivity(str(root / "dyads"))

    combined = _combine_results(root)
    settings = {
        "input_files": [str(path) for path in paths],
        "treatment_2015": "first_time_applicants (all first-time applicants)",
        "treatment_2022_non_ukrainian": "non-Ukrainian_first_time_applicant",
        "treatment_2022_ukrainian": "first_time_applicants_ukrainian",
        "model": "PanelOLS; dyad and period fixed effects; origin and mentioned exposures jointly",
        "default_se": "dyad clustered",
        "reported_effect": "sample-average discrete effect of +1 applicant per 1,000 residents",
        "source_files_edited": False,
        "run_time": datetime.now().isoformat(timespec="seconds"),
    }
    (root / "run_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Combined results:", combined)


if __name__ == "__main__":
    main()

# === BEGIN EMBEDDED sentiment_checks ===
#@# All DID robustness checks here
#@# ============================================================
#@# SCRIPT
#@# 1) Weighted DID
#@# 2) Clustering robustness
#@# 3) Raw linear-treatment sensitivity check
#@# 4) Placebo DID
#@# 5) Threshold +20 DID
#@# 6) Export master Excel summarizing significance
#@# 7) Export concise Excel summarizing parallel + DID significance + coefficients
#@#
#@# Output structure:
#@# robustness check/
#@#   ├── weighted/
#@#   ├── cluster_robustness/
#@#   │     ├── pair/
#@#   │     ├── origin_mentioned/
#@#   │     ├── origin/
#@#   │     └── mentioned/
#@#   ├── raw_treatment/
#@#   ├── placebo_test/
#@#   ├── threshold_plus20/
#@#   ├── robustness_significance_summary.xlsx
#@#   └── robustness_significance_summary_concise.xlsx
#@#
#@# Each folder outputs:
#@#   - results_2022crisis.csv
#@#   - results_2015crisis.csv
#@#   - figures/
#@# ============================================================
#@
#@import os
#@import re
#@import numpy as np
#@import pandas as pd
#@import matplotlib.pyplot as plt
#@from datetime import datetime
#@from linearmodels.panel import PanelOLS
#@from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator, FixedFormatter
#@
#@# ============================================================
#@# USER PATH SETTINGS
#@# ============================================================
#@INPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
#@OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "sentiment_hate_sensitivity", "checks"))
#@
#@combined_path = os.path.join(INPUT_ROOT, "pair_results_by_period.xlsx")
#@yearly_path = os.path.join(INPUT_ROOT, "pair_results_by_year.xlsx")
#@asylum_path = os.path.join(INPUT_ROOT, "asylum_data.xlsx")
#@
#@# ============================================================
#@# GENERAL SETTINGS
#@# ============================================================
#@INCLUDE_SELF = False
#@USE_WEIGHTS = False
#@
#@# Main specification used by the revised DID analysis.  All robustness checks
#@# except the explicitly named raw-treatment sensitivity check use log(1 + R),
#@# where R is first-time asylum applicants per 1,000 residents.
#@MAIN_TREATMENT_TRANSFORM = "log1p"
#@
#@# Report estimates in the reader-friendly original unit.  For log1p models,
#@# each coefficient is converted after estimation into the sample-average
#@# discrete effect of increasing R by 1 applicant per 1,000 residents.  The raw
#@# model coefficient is retained in separate CSV/Excel columns.
#@REPORT_AVERAGE_DISCRETE_EFFECT = True
#@DISCRETE_INCREASE_PER_1000 = 1.0
#@
#@# Original DID thresholds
#@DID2022_MIN_COUNT = 30
#@CRISIS2015_MIN_COUNT = 20
#@
#@# Threshold +20 robustness
#@THRESHOLD_PLUS20_MIN_COUNT_2022 = DID2022_MIN_COUNT + 20   # 50
#@THRESHOLD_PLUS20_MIN_COUNT_2015 = CRISIS2015_MIN_COUNT + 20  # 40
#@
#@# 2022 DID settings
#@DID2022_OUTCOME_SPECS = [
#@    {"ycol": "mean_sentiment_score"},
#@    {"ycol": "mean_hate_speech_score"},
#@]
#@DID2022_TREAT_SPECS = [
#@    {"treat_col": "non-Ukrainian_first_time_applicant"},
#@    {"treat_col": "first_time_applicants_ukrainian"},
#@]
#@
#@# 2015 crisis DID settings
#@CRISIS2015_OUTCOME_SPECS = [
#@    {"ycol": "mean_sentiment_score"},
#@    {"ycol": "mean_hate_speech_score"},
#@]
#@CRISIS2015_TREAT_SPECS = [
#@    {"treat_col": "first_time_applicants"},
#@]
#@
#@# ============================================================
#@# GLOBAL FIGURE STYLE
#@# ============================================================
#@plt.rcParams["font.family"] = "Arial"
#@plt.rcParams["font.size"] = 10
#@plt.rcParams["axes.unicode_minus"] = False
#@plt.rcParams["pdf.fonttype"] = 42
#@plt.rcParams["ps.fonttype"] = 42
#@
#@# ============================================================
#@# COLORS
#@# ============================================================
#@SENTIMENT_COLOR = "#0072B2"
#@HATE_COLOR = "#D55E00"
#@ZERO_LINE_COLOR = "#8A8A8A"
#@
#@# ============================================================
#@# FIGURE STYLE
#@# ============================================================
#@FIG_W = 4.8
#@FIG_H = 4.2
#@
#@MARKER_SIZE = 44
#@MARKER_EDGE_WIDTH = 1.2
#@
#@CAP_SIZE = 3.0
#@ERROR_LINE_WIDTH = 1.5
#@SPINE_WIDTH = 0.9
#@
#@TICK_SIZE = 9.5
#@LABEL_SIZE = 10.5
#@ANNOT_SIZE = 9.5
#@
#@# ============================================================
#@# OPTIONS
#@# ============================================================
#@SHOW_YLABEL = True
#@SHOW_POST_LABEL = True
#@SAVE_PDF = True
#@
#@# ============================================================
#@# SPECIAL FIGURES
#@# ============================================================
#@SPECIAL_OUTCOME = "mean_sentiment_score"
#@SPECIAL_TREATMENTS = {"first_time_applicants", "non-Ukrainian_first_time_applicant"}
#@SPECIAL_SIDE = "mentioned"
#@
#@SPECIAL_BOTTOM_LABEL = "-0.08"
#@SPECIAL_TOP_LABEL = "0.08"
#@
#@# Never overwrite genuine tick values with fixed labels after transforming the
#@# treatment scale or converting results to average discrete effects.
#@RELABEL_SPECIAL_YTICKS = False
#@
#@# ============================================================
#@# HELPERS
#@# ============================================================
#@def safe_mkdir(path: str):
#@    os.makedirs(path, exist_ok=True)
#@
#@
#@def now_tag():
#@    return datetime.now().strftime("%Y%m%d_%H%M%S")
#@
#@
#@def clean_country_name(x):
#@    if pd.isna(x):
#@        return x
#@    x = str(x).strip()
#@    x = re.sub(r"\s+", " ", x)
#@    x = x.lower()
#@    return x
#@
#@
#@def sanitize_filename(x):
#@    """
#@    Keep spaces and parentheses.
#@
#@    Only replace characters that are invalid in Windows filenames.
#@    """
#@    return (
#@        str(x)
#@        .replace("/", "_")
#@        .replace("\\", "_")
#@        .replace(":", "_")
#@        .replace("*", "_")
#@        .replace("?", "_")
#@        .replace('"', "_")
#@        .replace("<", "_")
#@        .replace(">", "_")
#@        .replace("|", "_")
#@    )
#@
#@def extract_term_or_nan(coef_df: pd.DataFrame, term: str):
#@    sub = coef_df[coef_df["term"] == term].copy()
#@    if sub.empty:
#@        return {
#@            "term": term,
#@            "coef": np.nan,
#@            "se": np.nan,
#@            "t": np.nan,
#@            "p": np.nan,
#@            "ci_low": np.nan,
#@            "ci_high": np.nan,
#@            "model_coef": np.nan,
#@            "model_se": np.nan,
#@            "model_ci_low": np.nan,
#@            "model_ci_high": np.nan,
#@            "average_discrete_factor": np.nan,
#@        }
#@    r = sub.iloc[0]
#@    return {
#@        "term": term,
#@        "coef": r["coef"],
#@        "se": r["se"],
#@        "t": r["t"],
#@        "p": r["p"],
#@        "ci_low": r["ci_low"],
#@        "ci_high": r["ci_high"],
#@        "model_coef": r.get("model_coef", r["coef"]),
#@        "model_se": r.get("model_se", r["se"]),
#@        "model_ci_low": r.get("model_ci_low", r["ci_low"]),
#@        "model_ci_high": r.get("model_ci_high", r["ci_high"]),
#@        "average_discrete_factor": r.get("average_discrete_factor", 1.0),
#@    }
#@
#@
#@def coef_table_from_res(res):
#@    return pd.DataFrame({
#@        "term": res.params.index,
#@        "coef": res.params.values,
#@        "se": res.std_errors.values,
#@        "t": res.tstats.values,
#@        "p": res.pvalues.values,
#@        "ci_low": res.conf_int().iloc[:, 0].values,
#@        "ci_high": res.conf_int().iloc[:, 1].values
#@    })
#@
#@
#@def _pair_level_treatment_values(panel: pd.DataFrame, raw_col: str):
#@    """Return one raw treatment rate per dyad after checking constancy."""
#@    if raw_col not in panel.columns:
#@        raise ValueError(f"Missing raw treatment column for reporting: {raw_col}")
#@
#@    variation = panel.groupby("pair")[raw_col].nunique(dropna=False)
#@    if (variation > 1).any():
#@        bad = variation[variation > 1].index.tolist()[:5]
#@        raise ValueError(
#@            f"Treatment is not constant within dyad for {raw_col}; examples: {bad}"
#@        )
#@
#@    values = (
#@        panel[["pair", raw_col]]
#@        .drop_duplicates("pair")[raw_col]
#@        .pipe(pd.to_numeric, errors="coerce")
#@        .dropna()
#@        .astype(float)
#@    )
#@    if values.empty:
#@        raise ValueError(f"No usable raw treatment values in {raw_col}")
#@    if (values < 0).any():
#@        raise ValueError(f"Negative raw treatment values in {raw_col}")
#@    return values
#@
#@
#@def average_discrete_factor(
#@    panel: pd.DataFrame,
#@    raw_col: str,
#@    treatment_transform: str,
#@):
#@    """
#@    Average change in the model treatment index when the original rate rises
#@    by DISCRETE_INCREASE_PER_1000 for every dyad in the estimation sample.
#@    """
#@    values = _pair_level_treatment_values(panel, raw_col)
#@    increase = float(DISCRETE_INCREASE_PER_1000)
#@    if increase <= 0:
#@        raise ValueError("DISCRETE_INCREASE_PER_1000 must be positive")
#@
#@    if treatment_transform == "log1p":
#@        contrasts = np.log1p(values + increase) - np.log1p(values)
#@    elif treatment_transform == "raw":
#@        contrasts = np.full(len(values), increase, dtype=float)
#@    else:
#@        raise ValueError(f"Unknown treatment_transform: {treatment_transform}")
#@
#@    return float(np.mean(contrasts))
#@
#@
#@def convert_coef_table_to_average_discrete_effect(
#@    coef_df: pd.DataFrame,
#@    panel: pd.DataFrame,
#@    term_to_raw_col: dict,
#@    treatment_transform: str,
#@):
#@    """
#@    Convert model-scale estimates into sample-average effects of +1 applicant
#@    per 1,000 residents.  Estimate, SE, and CI are multiplied by the same
#@    positive factor, so t statistics and p values remain unchanged.
#@    """
#@    out = coef_df.copy()
#@    out["model_coef"] = out["coef"]
#@    out["model_se"] = out["se"]
#@    out["model_ci_low"] = out["ci_low"]
#@    out["model_ci_high"] = out["ci_high"]
#@    out["average_discrete_factor"] = 1.0
#@
#@    if not REPORT_AVERAGE_DISCRETE_EFFECT:
#@        return out
#@
#@    for term, raw_col in term_to_raw_col.items():
#@        factor = average_discrete_factor(
#@            panel=panel,
#@            raw_col=raw_col,
#@            treatment_transform=treatment_transform,
#@        )
#@        mask = out["term"] == term
#@        if not mask.any():
#@            continue
#@        out.loc[mask, "average_discrete_factor"] = factor
#@        for col in ["coef", "se", "ci_low", "ci_high"]:
#@            out.loc[mask, col] = out.loc[mask, col] * factor
#@
#@    return out
#@
#@
#@def keep_common_pairs(panel: pd.DataFrame, periods):
#@    tmp = panel.groupby("pair")["period"].nunique()
#@    valid_pairs = tmp[tmp == len(periods)].index.tolist()
#@    return panel[panel["pair"].isin(valid_pairs)].copy(), valid_pairs
#@
#@
#@def get_line_color(outcome_value):
#@    if str(outcome_value) == "mean_sentiment_score":
#@        return SENTIMENT_COLOR
#@    return HATE_COLOR
#@
#@
#@def unified_x():
#@    return [0, 1, 2]
#@
#@
#@def unified_marker():
#@    return "o-"
#@
#@
#@def nice_ylim(ci_low, ci_high):
#@    vals = [v for v in ci_low + ci_high if pd.notna(v)]
#@    if not vals:
#@        return (-0.1, 0.1)
#@
#@    ymin = min(vals)
#@    ymax = max(vals)
#@
#@    ymin = min(ymin, 0)
#@    ymax = max(ymax, 0)
#@
#@    span = ymax - ymin
#@    pad = span * 0.18 if span > 0 else 0.05
#@
#@    return ymin - pad, ymax + pad
#@
#@
#@def ci_excludes_zero(ci_low, ci_high):
#@    if pd.isna(ci_low) or pd.isna(ci_high):
#@        return False
#@    return (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
#@
#@
#@def format_coef_label(value, ci_low=None, ci_high=None):
#@    """
#@    Display the reported average discrete effect, not the raw model beta.
#@    """
#@    if pd.isna(value):
#@        return ""
#@    return f"Effect = {value:.3f}"
#@
#@
#@def dynamic_formatter(x, pos):
#@    """
#@    Adapt tick precision to the magnitude of the axis.
#@
#@    The y-axis zero label is always displayed as 0.00.
#@    """
#@    if np.isclose(x, 0, atol=1e-12):
#@        return "0.00"
#@
#@    ax_abs = abs(x)
#@
#@    if ax_abs >= 1:
#@        return f"{x:.1f}"
#@    elif ax_abs >= 0.1:
#@        return f"{x:.2f}"
#@    else:
#@        return f"{x:.3f}"
#@
#@
#@def nice_ylim_raw(ci_low, ci_high):
#@    """
#@    Force y-axis to be symmetric around zero.
#@    """
#@    vals = [v for v in ci_low + ci_high if pd.notna(v)]
#@
#@    if not vals:
#@        return (-0.05, 0.05)
#@
#@    max_abs = max(abs(min(vals)), abs(max(vals)), 0)
#@    limit = max_abs * 1.22
#@
#@    min_half_span = 0.03
#@    if limit < min_half_span:
#@        limit = min_half_span
#@
#@    return -limit, limit
#@
#@
#@def style_axis(ax):
#@    ax.axhline(
#@        0,
#@        linestyle="--",
#@        linewidth=1.0,
#@        color=ZERO_LINE_COLOR,
#@        zorder=1
#@    )
#@
#@    ax.set_xlim(-0.30, 2.55)
#@
#@    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
#@    ax.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
#@
#@    ax.tick_params(
#@        axis="both",
#@        labelsize=TICK_SIZE,
#@        width=0.8,
#@        length=3.2,
#@        top=False,
#@        right=False
#@    )
#@
#@    ax.spines["top"].set_visible(False)
#@    ax.spines["right"].set_visible(False)
#@
#@    ax.spines["left"].set_linewidth(SPINE_WIDTH)
#@    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
#@
#@    ax.grid(False)
#@
#@
#@def display_prefix_for_dataset_type(dataset_type):
#@    """
#@    Convert internal dataset names to the figure-name prefixes used in the final DID script.
#@    """
#@    dataset_type = str(dataset_type)
#@
#@    if dataset_type == "did2022":
#@        return "2022"
#@    if dataset_type == "crisis2015":
#@        return "crisis2015"
#@
#@    return dataset_type
#@
#@
#@def use_special_tick_labels(row, side, prefix):
#@    """
#@    Special figures:
#@    1. crisis2015_sentiment_first-time asylum applicant_mentioned.png
#@    2. 2022_sentiment_first-time asylum applicant_mentioned.png
#@    """
#@    outcome = str(row["outcome"])
#@    treatment = str(row["treatment"])
#@    prefix = display_prefix_for_dataset_type(prefix)
#@
#@    cond1 = (
#@        prefix == "crisis2015"
#@        and outcome == SPECIAL_OUTCOME
#@        and treatment in SPECIAL_TREATMENTS
#@        and side == SPECIAL_SIDE
#@    )
#@
#@    cond2 = (
#@        prefix == "2022"
#@        and outcome == SPECIAL_OUTCOME
#@        and treatment in SPECIAL_TREATMENTS
#@        and side == SPECIAL_SIDE
#@    )
#@
#@    return cond1 or cond2
#@
#@
#@def relabel_top_bottom_yticks(ax, bottom_label="-0.08", top_label="0.08"):
#@    ymin, ymax = ax.get_ylim()
#@    ticks = ax.get_yticks()
#@    visible_ticks = [t for t in ticks if ymin <= t <= ymax]
#@
#@    if len(visible_ticks) < 2:
#@        return
#@
#@    bottom_tick = visible_ticks[0]
#@    top_tick = visible_ticks[-1]
#@
#@    labels = []
#@    for t in visible_ticks:
#@        if np.isclose(t, bottom_tick):
#@            labels.append(bottom_label)
#@        elif np.isclose(t, top_tick):
#@            labels.append(top_label)
#@        else:
#@            labels.append(dynamic_formatter(t, None))
#@
#@    ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
#@    ax.yaxis.set_major_formatter(FixedFormatter(labels))
#@
#@
#@def draw_coef_point(
#@    ax,
#@    x_value,
#@    y_value,
#@    yerr_lower,
#@    yerr_upper,
#@    ci_low,
#@    ci_high,
#@    color,
#@    zorder=4
#@):
#@    """
#@    Draw CI and coefficient point separately.
#@
#@    All estimates use the same filled marker.  The confidence interval conveys
#@    uncertainty without redundant significance stars or hollow markers.
#@    """
#@    if pd.isna(y_value):
#@        return
#@
#@    if pd.notna(yerr_lower) and pd.notna(yerr_upper):
#@        ax.errorbar(
#@            [x_value],
#@            [y_value],
#@            yerr=[[yerr_lower], [yerr_upper]],
#@            fmt="none",
#@            ecolor=color,
#@            elinewidth=ERROR_LINE_WIDTH,
#@            capsize=0,
#@            alpha=1.0,
#@            zorder=zorder
#@        )
#@
#@    ax.scatter(
#@        [x_value],
#@        [y_value],
#@        s=MARKER_SIZE,
#@        facecolors=color,
#@        edgecolors=color,
#@        linewidths=MARKER_EDGE_WIDTH,
#@        alpha=1.0,
#@        zorder=zorder + 1
#@    )
#@
#@
#@def outcome_to_filename_label(outcome):
#@    outcome = str(outcome)
#@
#@    if outcome == "mean_sentiment_score":
#@        return "sentiment"
#@    elif outcome == "mean_hate_speech_score":
#@        return "hate speech"
#@
#@    return sanitize_filename(outcome)
#@
#@
#@def treatment_to_filename_label(treatment):
#@    treatment = str(treatment)
#@
#@    if treatment == "non-Ukrainian_first_time_applicant":
#@        return "non-Ukrainian first-time asylum applicant"
#@    if treatment == "first_time_applicants":
#@        return "first-time asylum applicant"
#@    elif treatment == "first_time_applicants_ukrainian":
#@        return "first-time asylum applicant (Ukrainians)"
#@
#@    return sanitize_filename(treatment)
#@
#@
#@def make_figure_filename(prefix, outcome, treatment, side):
#@    """
#@    Examples:
#@    2022_hate speech_first-time asylum applicant (Ukrainians)_origin.png
#@    crisis2015_sentiment_first-time asylum applicant_mentioned.png
#@    """
#@    crisis_label = display_prefix_for_dataset_type(prefix)
#@    outcome_label = outcome_to_filename_label(outcome)
#@    treatment_label = treatment_to_filename_label(treatment)
#@    side_label = str(side)
#@
#@    fname = f"{crisis_label}_{outcome_label}_{treatment_label}_{side_label}.png"
#@    return sanitize_filename(fname)
#@
#@
#@
#@def make_failed_summary_row_generic(
#@    dataset_type,
#@    ycol,
#@    treat_col,
#@    error_msg,
#@    treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@):
#@    return {
#@        "dataset_type": dataset_type,
#@        "outcome": ycol,
#@        "treatment": treat_col,
#@        "treatment_transform": treatment_transform,
#@        "model_treatment_scale": (
#@            "log(1 + first-time asylum applicants per 1,000 residents)"
#@            if treatment_transform == "log1p"
#@            else "first-time asylum applicants per 1,000 residents"
#@        ),
#@        "reported_effect_scale": (
#@            "sample-average discrete effect of +1 applicant per 1,000 residents"
#@        ),
#@
#@        "n_pairs_pre_total": np.nan,
#@        "n_obs_pre": np.nan,
#@        "n_pairs_post_common": np.nan,
#@        "n_obs_post": np.nan,
#@
#@        "origin_parallel": "failed",
#@        "mentioned_parallel": "failed",
#@        "overall_parallel": "failed",
#@
#@        "origin_pre_period": np.nan,
#@        "origin_pre_event_time": -2,
#@        "origin_pre_coef": np.nan,
#@        "origin_pre_model_coef": np.nan,
#@        "origin_pre_average_discrete_factor": np.nan,
#@        "origin_pre_se": np.nan,
#@        "origin_pre_model_se": np.nan,
#@        "origin_pre_p": np.nan,
#@        "origin_pre_ci_low": np.nan,
#@        "origin_pre_ci_high": np.nan,
#@        "origin_pre_model_ci_low": np.nan,
#@        "origin_pre_model_ci_high": np.nan,
#@
#@        "origin_base_period": np.nan,
#@        "origin_base_event_time": -1,
#@        "origin_base_coef": 0.0,
#@        "origin_base_se": 0.0,
#@        "origin_base_p": np.nan,
#@        "origin_base_ci_low": 0.0,
#@        "origin_base_ci_high": 0.0,
#@
#@        "origin_post_period": np.nan,
#@        "origin_post_event_time": 1,
#@        "origin_post_coef": np.nan,
#@        "origin_post_model_coef": np.nan,
#@        "origin_post_average_discrete_factor": np.nan,
#@        "origin_post_se": np.nan,
#@        "origin_post_model_se": np.nan,
#@        "origin_post_p": np.nan,
#@        "origin_post_ci_low": np.nan,
#@        "origin_post_ci_high": np.nan,
#@        "origin_post_model_ci_low": np.nan,
#@        "origin_post_model_ci_high": np.nan,
#@
#@        "mentioned_pre_period": np.nan,
#@        "mentioned_pre_event_time": -2,
#@        "mentioned_pre_coef": np.nan,
#@        "mentioned_pre_model_coef": np.nan,
#@        "mentioned_pre_average_discrete_factor": np.nan,
#@        "mentioned_pre_se": np.nan,
#@        "mentioned_pre_model_se": np.nan,
#@        "mentioned_pre_p": np.nan,
#@        "mentioned_pre_ci_low": np.nan,
#@        "mentioned_pre_ci_high": np.nan,
#@        "mentioned_pre_model_ci_low": np.nan,
#@        "mentioned_pre_model_ci_high": np.nan,
#@
#@        "mentioned_base_period": np.nan,
#@        "mentioned_base_event_time": -1,
#@        "mentioned_base_coef": 0.0,
#@        "mentioned_base_se": 0.0,
#@        "mentioned_base_p": np.nan,
#@        "mentioned_base_ci_low": 0.0,
#@        "mentioned_base_ci_high": 0.0,
#@
#@        "mentioned_post_period": np.nan,
#@        "mentioned_post_event_time": 1,
#@        "mentioned_post_coef": np.nan,
#@        "mentioned_post_model_coef": np.nan,
#@        "mentioned_post_average_discrete_factor": np.nan,
#@        "mentioned_post_se": np.nan,
#@        "mentioned_post_model_se": np.nan,
#@        "mentioned_post_p": np.nan,
#@        "mentioned_post_ci_low": np.nan,
#@        "mentioned_post_ci_high": np.nan,
#@        "mentioned_post_model_ci_low": np.nan,
#@        "mentioned_post_model_ci_high": np.nan,
#@
#@        "estimation_status": "failed",
#@        "error_message": str(error_msg),
#@    }
#@
#@
#@# ============================================================
#@# LOAD PERIOD PANEL
#@# ============================================================
#@def read_panel_from_period_file(combined_path: str, sheets, ycol: str, min_count: int, include_self: bool):
#@    data_dict = pd.read_excel(combined_path, sheet_name=sheets)
#@    all_rows = []
#@
#@    for sh in sheets:
#@        df = data_dict[sh].copy()
#@
#@        need_cols = ["origin_country", "mentioned_country", "post_count", ycol]
#@        missing = [c for c in need_cols if c not in df.columns]
#@        if missing:
#@            raise ValueError(f"[{sh}] missing columns: {missing}")
#@
#@        df["origin_country"] = df["origin_country"].map(clean_country_name)
#@        df["mentioned_country"] = df["mentioned_country"].map(clean_country_name)
#@        df["post_count"] = pd.to_numeric(df["post_count"], errors="coerce")
#@        df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
#@
#@        df = df[df["post_count"] >= min_count].copy()
#@        df = df[~df[ycol].isna()].copy()
#@
#@        df = df.rename(columns={
#@            "origin_country": "origin",
#@            "mentioned_country": "mentioned",
#@            ycol: "Y"
#@        })
#@
#@        if not include_self:
#@            df = df[df["origin"] != df["mentioned"]].copy()
#@
#@        df["period"] = sh
#@        df["pair"] = df["origin"] + "->" + df["mentioned"]
#@        df = df.drop_duplicates(subset=["pair", "period"])
#@
#@        all_rows.append(df[["pair", "origin", "mentioned", "period", "Y", "post_count"]])
#@
#@    panel = pd.concat(all_rows, ignore_index=True)
#@    return panel
#@
#@
#@# ============================================================
#@# YEARLY FILE HELPERS
#@# ============================================================
#@def _find_year_col(df):
#@    for c in ["Year", "year", "YEAR"]:
#@        if c in df.columns:
#@            return c
#@    return None
#@
#@
#@def _standardize_yearly_df(df: pd.DataFrame, year_value=None):
#@    df = df.copy()
#@
#@    need_some_country_cols = ["origin_country", "mentioned_country"]
#@    missing_country = [c for c in need_some_country_cols if c not in df.columns]
#@    if missing_country:
#@        raise ValueError(f"Yearly data missing columns: {missing_country}")
#@
#@    df["origin_country"] = df["origin_country"].map(clean_country_name)
#@    df["mentioned_country"] = df["mentioned_country"].map(clean_country_name)
#@
#@    needed_numeric = [
#@        "positive_post_count", "neutral_post_count", "negative_post_count",
#@        "hate_speech_post_count", "normal_post_count", "offensive_post_count"
#@    ]
#@    missing_numeric = [c for c in needed_numeric if c not in df.columns]
#@    if missing_numeric:
#@        raise ValueError(f"Yearly data missing columns: {missing_numeric}")
#@
#@    for c in needed_numeric:
#@        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
#@
#@    if year_value is not None:
#@        df["year"] = int(year_value)
#@    else:
#@        yc = _find_year_col(df)
#@        if yc is None:
#@            raise ValueError("Yearly file must have per-year sheets or a Year/year column.")
#@        df["year"] = pd.to_numeric(df[yc], errors="coerce")
#@
#@    df = df.dropna(subset=["year"]).copy()
#@    df["year"] = df["year"].astype(int)
#@
#@    return df
#@
#@
#@def load_yearly_long(yearly_path: str, target_years):
#@    xls = pd.ExcelFile(yearly_path)
#@    sheet_names = xls.sheet_names
#@
#@    year_sheets_found = []
#@    for sh in sheet_names:
#@        s = str(sh).strip()
#@        if s.isdigit() and int(s) in target_years:
#@            year_sheets_found.append(int(s))
#@
#@    if set(year_sheets_found) == set(target_years):
#@        parts = []
#@        for yy in target_years:
#@            tmp = pd.read_excel(yearly_path, sheet_name=str(yy))
#@            tmp = _standardize_yearly_df(tmp, year_value=yy)
#@            parts.append(tmp)
#@        out = pd.concat(parts, ignore_index=True)
#@        return out
#@
#@    for sh in sheet_names:
#@        tmp = pd.read_excel(yearly_path, sheet_name=sh)
#@        yc = _find_year_col(tmp)
#@        needed = [
#@            "origin_country", "mentioned_country",
#@            "positive_post_count", "neutral_post_count", "negative_post_count",
#@            "hate_speech_post_count", "normal_post_count", "offensive_post_count"
#@        ]
#@        if yc is not None and all(c in tmp.columns for c in needed):
#@            tmp = _standardize_yearly_df(tmp, year_value=None)
#@            tmp = tmp[tmp["year"].isin(target_years)].copy()
#@            return tmp
#@
#@    raise ValueError("Cannot parse yearly file correctly.")
#@
#@
#@# ============================================================
#@# GENERIC YEARLY AGGREGATOR FOR FLEXIBLE PLACEBO 2015
#@# ============================================================
#@def build_yearly_period_panel_from_mapping(yearly_path: str, target_years, period_map_func,
#@                                           ycol: str, min_count: int, include_self: bool):
#@    df = load_yearly_long(yearly_path, target_years).copy()
#@    df = df.rename(columns={"origin_country": "origin", "mentioned_country": "mentioned"})
#@
#@    if not include_self:
#@        df = df[df["origin"] != df["mentioned"]].copy()
#@
#@    df["pair"] = df["origin"] + "->" + df["mentioned"]
#@    df["period"] = df["year"].map(period_map_func)
#@    df = df.dropna(subset=["period"]).copy()
#@
#@    agg = (
#@        df.groupby(["pair", "origin", "mentioned", "period"], as_index=False)
#@        .agg({
#@            "positive_post_count": "sum",
#@            "neutral_post_count": "sum",
#@            "negative_post_count": "sum",
#@            "hate_speech_post_count": "sum",
#@            "normal_post_count": "sum",
#@            "offensive_post_count": "sum",
#@            "year": "nunique"
#@        })
#@        .rename(columns={"year": "n_years"})
#@    )
#@
#@    sent_den = agg["positive_post_count"] + agg["neutral_post_count"] + agg["negative_post_count"]
#@    hate_den = agg["hate_speech_post_count"] + agg["normal_post_count"] + agg["offensive_post_count"]
#@
#@    if ycol == "mean_sentiment_score":
#@        agg["raw_num"] = agg["positive_post_count"] - agg["negative_post_count"]
#@        agg["raw_den"] = sent_den
#@        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
#@        agg["post_count"] = agg["raw_den"]
#@    elif ycol == "mean_hate_speech_score":
#@        agg["raw_num"] = agg["hate_speech_post_count"] * 2 + agg["offensive_post_count"]
#@        agg["raw_den"] = hate_den
#@        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
#@        agg["post_count"] = agg["raw_den"]
#@    else:
#@        raise ValueError(f"Unsupported ycol: {ycol}")
#@
#@    agg = agg[agg["post_count"] >= min_count].copy()
#@    agg = agg[~agg["Y"].isna()].copy()
#@
#@    return agg[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den", "n_years"]]
#@
#@
#@# ============================================================
#@# BUILD PRE PANEL - 2022 DID VERSION
#@# ============================================================
#@def _map_2022_pre_period(y):
#@    y = int(y)
#@    if 2010 <= y <= 2014:
#@        return "2010-2014"
#@    elif 2015 <= y <= 2021:
#@        return "2015-2021"
#@    return np.nan
#@
#@
#@def build_pre_panel_2022_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
#@    target_years = list(range(2010, 2022))
#@    return build_yearly_period_panel_from_mapping(
#@        yearly_path=yearly_path,
#@        target_years=target_years,
#@        period_map_func=_map_2022_pre_period,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@
#@
#@# ============================================================
#@# BUILD PRE PANEL - 2015 CRISIS VERSION
#@# ============================================================
#@def _map_2015_pre_period(y):
#@    y = int(y)
#@    if y in [2010, 2011, 2012]:
#@        return "2010-2012"
#@    elif y in [2013, 2014]:
#@        return "2013-2014"
#@    return np.nan
#@
#@
#@def build_pre_panel_2015_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
#@    target_years = [2010, 2011, 2012, 2013, 2014]
#@    return build_yearly_period_panel_from_mapping(
#@        yearly_path=yearly_path,
#@        target_years=target_years,
#@        period_map_func=_map_2015_pre_period,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@
#@
#@# ============================================================
#@# PLACEBO BUILDERS - 2022
#@# ============================================================
#@def _map_placebo_2022_pre(y):
#@    y = int(y)
#@    if 2010 <= y <= 2012:
#@        return "2010-2012"
#@    elif 2013 <= y <= 2014:
#@        return "2013-2014"
#@    return np.nan
#@
#@
#@def _map_placebo_2022_post(y):
#@    y = int(y)
#@    if 2013 <= y <= 2014:
#@        return "2013-2014"
#@    elif 2015 <= y <= 2021:
#@        return "2015-2021"
#@    return np.nan
#@
#@
#@def build_placebo_2022_pre_panel_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
#@    target_years = [2010, 2011, 2012, 2013, 2014]
#@    return build_yearly_period_panel_from_mapping(
#@        yearly_path=yearly_path,
#@        target_years=target_years,
#@        period_map_func=_map_placebo_2022_pre,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@
#@
#@def build_placebo_2022_post_panel_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
#@    target_years = list(range(2013, 2022))
#@    return build_yearly_period_panel_from_mapping(
#@        yearly_path=yearly_path,
#@        target_years=target_years,
#@        period_map_func=_map_placebo_2022_post,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@
#@
#@# ============================================================
#@# PLACEBO 2015: flexible fallback splits
#@# ============================================================
#@PLACEBO_2015_SPECS = [
#@    {
#@        "name": "v1_standard_wider",
#@        "pre_target_years": [2010, 2011, 2012, 2013],
#@        "post_target_years": [2012, 2013, 2014],
#@        "pre_periods": ["2010-2011", "2012-2013"],
#@        "post_periods": ["2012-2013", "2014"],
#@        "period_labels": {
#@            "origin_pre_period": "2010-2011",
#@            "origin_base_period": "2012-2013",
#@            "origin_post_period": "2014",
#@            "mentioned_pre_period": "2010-2011",
#@            "mentioned_base_period": "2012-2013",
#@            "mentioned_post_period": "2014",
#@        }
#@    },
#@    {
#@        "name": "v2_widest",
#@        "pre_target_years": [2010, 2011, 2012, 2013, 2014],
#@        "post_target_years": [2011, 2012, 2013, 2014],
#@        "pre_periods": ["2010-2012", "2013-2014"],
#@        "post_periods": ["2011-2012", "2013-2014"],
#@        "period_labels": {
#@            "origin_pre_period": "2010-2012",
#@            "origin_base_period": "2013-2014",
#@            "origin_post_period": "2013-2014",
#@            "mentioned_pre_period": "2010-2012",
#@            "mentioned_base_period": "2013-2014",
#@            "mentioned_post_period": "2013-2014",
#@        }
#@    },
#@    {
#@        "name": "v3_shifted",
#@        "pre_target_years": [2010, 2011, 2012, 2013],
#@        "post_target_years": [2011, 2012, 2013, 2014],
#@        "pre_periods": ["2010-2012", "2013"],
#@        "post_periods": ["2011-2012", "2013-2014"],
#@        "period_labels": {
#@            "origin_pre_period": "2010-2012",
#@            "origin_base_period": "2013",
#@            "origin_post_period": "2013-2014",
#@            "mentioned_pre_period": "2010-2012",
#@            "mentioned_base_period": "2013",
#@            "mentioned_post_period": "2013-2014",
#@        }
#@    }
#@]
#@
#@
#@def placebo2015_pre_mapper_factory(spec_name):
#@    def mapper(y):
#@        y = int(y)
#@        if spec_name == "v1_standard_wider":
#@            if y in [2010, 2011]:
#@                return "2010-2011"
#@            elif y in [2012, 2013]:
#@                return "2012-2013"
#@        elif spec_name == "v2_widest":
#@            if y in [2010, 2011, 2012]:
#@                return "2010-2012"
#@            elif y in [2013, 2014]:
#@                return "2013-2014"
#@        elif spec_name == "v3_shifted":
#@            if y in [2010, 2011, 2012]:
#@                return "2010-2012"
#@            elif y in [2013]:
#@                return "2013"
#@        return np.nan
#@    return mapper
#@
#@
#@def placebo2015_post_mapper_factory(spec_name):
#@    def mapper(y):
#@        y = int(y)
#@        if spec_name == "v1_standard_wider":
#@            if y in [2012, 2013]:
#@                return "2012-2013"
#@            elif y in [2014]:
#@                return "2014"
#@        elif spec_name == "v2_widest":
#@            if y in [2011, 2012]:
#@                return "2011-2012"
#@            elif y in [2013, 2014]:
#@                return "2013-2014"
#@        elif spec_name == "v3_shifted":
#@            if y in [2011, 2012]:
#@                return "2011-2012"
#@            elif y in [2013, 2014]:
#@                return "2013-2014"
#@        return np.nan
#@    return mapper
#@
#@
#@# ============================================================
#@# TREATMENT HELPERS
#@# ============================================================
#@def make_treat_k(
#@    asylum_path: str,
#@    year: int,
#@    treat_col: str,
#@    treatment_transform: str = MAIN_TREATMENT_TRANSFORM,
#@):
#@    a = pd.read_excel(asylum_path, sheet_name=str(year)).copy()
#@
#@    need_cols = ["host_country", treat_col, "population"]
#@    missing = [c for c in need_cols if c not in a.columns]
#@    if missing:
#@        raise ValueError(f"[asylum {year}] missing columns: {missing}")
#@
#@    a = a.rename(columns={"host_country": "country"})
#@    a["country"] = a["country"].map(clean_country_name)
#@
#@    a[treat_col] = pd.to_numeric(a[treat_col], errors="coerce")
#@    a["population"] = pd.to_numeric(a["population"], errors="coerce")
#@
#@    treat_k_col = f"treat_k_{year}"
#@    raw_treat_k_col = f"raw_treat_k_{year}"
#@    raw_rate = 1000.0 * a[treat_col] / a["population"]
#@    a[raw_treat_k_col] = raw_rate
#@
#@    if (raw_rate.dropna() < 0).any():
#@        raise ValueError(f"[{treat_col}, {year}] negative treatment rate found")
#@
#@    if treatment_transform == "log1p":
#@        a[treat_k_col] = np.log1p(raw_rate)
#@    elif treatment_transform == "raw":
#@        a[treat_k_col] = raw_rate
#@    else:
#@        raise ValueError(f"Unknown treatment_transform: {treatment_transform}")
#@
#@    return a[["country", treat_k_col, raw_treat_k_col]]
#@
#@
#@def attach_treatment_two_sides(
#@    panel: pd.DataFrame,
#@    asylum_path: str,
#@    treat_col: str,
#@    treat_year: int,
#@    treatment_transform: str = MAIN_TREATMENT_TRANSFORM,
#@):
#@    t = make_treat_k(
#@        asylum_path,
#@        treat_year,
#@        treat_col=treat_col,
#@        treatment_transform=treatment_transform,
#@    )
#@    tk = f"treat_k_{treat_year}"
#@    raw_tk = f"raw_treat_k_{treat_year}"
#@
#@    panel = panel.merge(
#@        t.rename(columns={
#@            "country": "origin",
#@            tk: "treat_origin",
#@            raw_tk: "raw_treat_origin",
#@        }),
#@        on="origin",
#@        how="left"
#@    )
#@    panel = panel.merge(
#@        t.rename(columns={
#@            "country": "mentioned",
#@            tk: "treat_mentioned",
#@            raw_tk: "raw_treat_mentioned",
#@        }),
#@        on="mentioned",
#@        how="left"
#@    )
#@
#@    panel["treat_origin"] = pd.to_numeric(panel["treat_origin"], errors="coerce")
#@    panel["treat_mentioned"] = pd.to_numeric(panel["treat_mentioned"], errors="coerce")
#@    panel["raw_treat_origin"] = pd.to_numeric(panel["raw_treat_origin"], errors="coerce")
#@    panel["raw_treat_mentioned"] = pd.to_numeric(panel["raw_treat_mentioned"], errors="coerce")
#@    return panel
#@
#@
#@def attach_dyadic_treat_k_2015(
#@    panel: pd.DataFrame,
#@    asylum_path: str,
#@    treat_col: str,
#@    treatment_transform: str = MAIN_TREATMENT_TRANSFORM,
#@):
#@    t15 = make_treat_k(
#@        asylum_path,
#@        2015,
#@        treat_col=treat_col,
#@        treatment_transform=treatment_transform,
#@    )
#@
#@    panel = panel.merge(
#@        t15.rename(columns={
#@            "country": "origin",
#@            "treat_k_2015": "treat_origin_2015",
#@            "raw_treat_k_2015": "raw_treat_origin_2015",
#@        }),
#@        on="origin", how="left"
#@    )
#@    panel = panel.merge(
#@        t15.rename(columns={
#@            "country": "mentioned",
#@            "treat_k_2015": "treat_mentioned_2015",
#@            "raw_treat_k_2015": "raw_treat_mentioned_2015",
#@        }),
#@        on="mentioned", how="left"
#@    )
#@
#@    panel["treat_origin_2015"] = pd.to_numeric(panel["treat_origin_2015"], errors="coerce")
#@    panel["treat_mentioned_2015"] = pd.to_numeric(panel["treat_mentioned_2015"], errors="coerce")
#@    panel["raw_treat_origin_2015"] = pd.to_numeric(panel["raw_treat_origin_2015"], errors="coerce")
#@    panel["raw_treat_mentioned_2015"] = pd.to_numeric(panel["raw_treat_mentioned_2015"], errors="coerce")
#@    return panel
#@
#@
#@# ============================================================
#@# VARIABLE BUILDERS
#@# ============================================================
#@def build_pretrend_vars_2022(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2010-2014": 0, "2015-2021": 1})
#@    panel["PrePeriod"] = (panel["period"] == "2010-2014").astype(int)
#@    panel["PRE_origin"] = panel["treat_origin"] * panel["PrePeriod"]
#@    panel["PRE_mentioned"] = panel["treat_mentioned"] * panel["PrePeriod"]
#@    return panel
#@
#@
#@def build_post_did_vars_2022(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2015-2021": 0, "2022-2024": 1})
#@    panel["PostPeriod"] = (panel["period"] == "2022-2024").astype(int)
#@    panel["POST_origin"] = panel["treat_origin"] * panel["PostPeriod"]
#@    panel["POST_mentioned"] = panel["treat_mentioned"] * panel["PostPeriod"]
#@    return panel
#@
#@
#@def build_pretrend_vars_2015(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2010-2012": 0, "2013-2014": 1})
#@    panel["PrePeriod"] = (panel["period"] == "2010-2012").astype(int)
#@    panel["PRE_origin"] = panel["treat_origin_2015"] * panel["PrePeriod"]
#@    panel["PRE_mentioned"] = panel["treat_mentioned_2015"] * panel["PrePeriod"]
#@    return panel
#@
#@
#@def build_post_did_vars_2015(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2013-2014": 0, "2015-2021": 1})
#@    panel["Post2015"] = (panel["period"] == "2015-2021").astype(int)
#@    panel["POST_origin"] = panel["treat_origin_2015"] * panel["Post2015"]
#@    panel["POST_mentioned"] = panel["treat_mentioned_2015"] * panel["Post2015"]
#@    return panel
#@
#@
#@def build_pretrend_vars_placebo_2022(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2010-2012": 0, "2013-2014": 1})
#@    panel["PrePeriod"] = (panel["period"] == "2010-2012").astype(int)
#@    panel["PRE_origin"] = panel["treat_origin"] * panel["PrePeriod"]
#@    panel["PRE_mentioned"] = panel["treat_mentioned"] * panel["PrePeriod"]
#@    return panel
#@
#@
#@def build_post_did_vars_placebo_2022(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2013-2014": 0, "2015-2021": 1})
#@    panel["PlaceboPost"] = (panel["period"] == "2015-2021").astype(int)
#@    panel["POST_origin"] = panel["treat_origin"] * panel["PlaceboPost"]
#@    panel["POST_mentioned"] = panel["treat_mentioned"] * panel["PlaceboPost"]
#@    return panel
#@
#@
#@def build_pretrend_vars_placebo_2015(panel: pd.DataFrame, pre_period, base_period):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({pre_period: 0, base_period: 1})
#@    panel["PrePeriod"] = (panel["period"] == pre_period).astype(int)
#@    panel["PRE_origin"] = panel["treat_origin_2015"] * panel["PrePeriod"]
#@    panel["PRE_mentioned"] = panel["treat_mentioned_2015"] * panel["PrePeriod"]
#@    return panel
#@
#@
#@def build_post_did_vars_placebo_2015(panel: pd.DataFrame, base_period, post_period):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({base_period: 0, post_period: 1})
#@    panel["PlaceboPost"] = (panel["period"] == post_period).astype(int)
#@    panel["POST_origin"] = panel["treat_origin_2015"] * panel["PlaceboPost"]
#@    panel["POST_mentioned"] = panel["treat_mentioned_2015"] * panel["PlaceboPost"]
#@    return panel
#@
#@
#@# ============================================================
#@# REGRESSION
#@# ============================================================
#@def run_panel_ols(panel: pd.DataFrame, xcols, cluster_mode: str = "pair", use_weights: bool = False):
#@    df = panel.copy().set_index(["pair", "period_id"]).sort_index()
#@
#@    needed_cols = ["Y", "post_count", "origin", "mentioned"] + xcols
#@    df = df.dropna(subset=needed_cols).copy()
#@
#@    if df.empty:
#@        raise ValueError("Estimation sample is empty after dropping missing values.")
#@    if df.index.get_level_values(0).nunique() < 2:
#@        raise ValueError("Fewer than 2 unique pairs in estimation sample.")
#@    if df.index.get_level_values(1).nunique() < 2:
#@        raise ValueError("Fewer than 2 time periods in estimation sample.")
#@
#@    y = df["Y"]
#@    X = df[xcols].copy()
#@
#@    keep_cols = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
#@    X = X[keep_cols]
#@
#@    if X.shape[1] == 0:
#@        raise ValueError("No regressor with variation remains after filtering.")
#@
#@    model_kwargs = dict(
#@        dependent=y,
#@        exog=X,
#@        entity_effects=True,
#@        time_effects=True,
#@        drop_absorbed=True
#@    )
#@
#@    if use_weights:
#@        model_kwargs["weights"] = pd.to_numeric(df["post_count"], errors="coerce").astype(float)
#@
#@    model = PanelOLS(**model_kwargs)
#@
#@    if cluster_mode == "pair":
#@        try:
#@            res = model.fit(cov_type="clustered", cluster_entity=True)
#@        except ZeroDivisionError:
#@            raise ValueError("Zero degrees of freedom in clustered covariance.")
#@    elif cluster_mode == "origin_mentioned":
#@        clusters = pd.DataFrame({
#@            "origin": df["origin"].astype("category").cat.codes,
#@            "mentioned": df["mentioned"].astype("category").cat.codes
#@        }, index=df.index)
#@        try:
#@            res = model.fit(cov_type="clustered", clusters=clusters)
#@        except ZeroDivisionError:
#@            raise ValueError("Zero degrees of freedom in clustered covariance.")
#@    elif cluster_mode == "origin":
#@        clusters = pd.Series(df["origin"].astype("category").cat.codes, index=df.index)
#@        try:
#@            res = model.fit(cov_type="clustered", clusters=clusters)
#@        except ZeroDivisionError:
#@            raise ValueError("Zero degrees of freedom in clustered covariance.")
#@    elif cluster_mode == "mentioned":
#@        clusters = pd.Series(df["mentioned"].astype("category").cat.codes, index=df.index)
#@        try:
#@            res = model.fit(cov_type="clustered", clusters=clusters)
#@        except ZeroDivisionError:
#@            raise ValueError("Zero degrees of freedom in clustered covariance.")
#@    else:
#@        raise ValueError("Unknown cluster_mode")
#@
#@    return res, df
#@
#@
#@# ============================================================
#@# FINAL SUMMARY ROWS
#@# ============================================================
#@def make_final_summary_row_common(
#@    dataset_type, ycol, treat_col,
#@    pre_coef_df, post_coef_df, pre_panel, post_panel,
#@    periods, treatment_transform
#@):
#@    pre_origin = extract_term_or_nan(pre_coef_df, "PRE_origin")
#@    pre_mentioned = extract_term_or_nan(pre_coef_df, "PRE_mentioned")
#@    post_origin = extract_term_or_nan(post_coef_df, "POST_origin")
#@    post_mentioned = extract_term_or_nan(post_coef_df, "POST_mentioned")
#@
#@    origin_parallel = "pass" if pd.notna(pre_origin["p"]) and pre_origin["p"] >= 0.05 else "fail"
#@    mentioned_parallel = "pass" if pd.notna(pre_mentioned["p"]) and pre_mentioned["p"] >= 0.05 else "fail"
#@    overall_parallel = "pass_both_sides" if origin_parallel == "pass" and mentioned_parallel == "pass" else "fail_at_least_one_side"
#@
#@    return {
#@        "dataset_type": dataset_type,
#@        "outcome": ycol,
#@        "treatment": treat_col,
#@        "treatment_transform": treatment_transform,
#@        "model_treatment_scale": (
#@            "log(1 + first-time asylum applicants per 1,000 residents)"
#@            if treatment_transform == "log1p"
#@            else "first-time asylum applicants per 1,000 residents"
#@        ),
#@        "reported_effect_scale": (
#@            "sample-average discrete effect of +1 applicant per 1,000 residents"
#@        ),
#@
#@        "n_pairs_pre_total": pre_panel["pair"].nunique(),
#@        "n_obs_pre": len(pre_panel),
#@        "n_pairs_post_common": post_panel["pair"].nunique(),
#@        "n_obs_post": len(post_panel),
#@
#@        "origin_parallel": origin_parallel,
#@        "mentioned_parallel": mentioned_parallel,
#@        "overall_parallel": overall_parallel,
#@
#@        "origin_pre_period": periods["origin_pre_period"],
#@        "origin_pre_event_time": -2,
#@        "origin_pre_coef": pre_origin["coef"],
#@        "origin_pre_model_coef": pre_origin["model_coef"],
#@        "origin_pre_average_discrete_factor": pre_origin["average_discrete_factor"],
#@        "origin_pre_se": pre_origin["se"],
#@        "origin_pre_model_se": pre_origin["model_se"],
#@        "origin_pre_p": pre_origin["p"],
#@        "origin_pre_ci_low": pre_origin["ci_low"],
#@        "origin_pre_ci_high": pre_origin["ci_high"],
#@        "origin_pre_model_ci_low": pre_origin["model_ci_low"],
#@        "origin_pre_model_ci_high": pre_origin["model_ci_high"],
#@
#@        "origin_base_period": periods["origin_base_period"],
#@        "origin_base_event_time": -1,
#@        "origin_base_coef": 0.0,
#@        "origin_base_se": 0.0,
#@        "origin_base_p": np.nan,
#@        "origin_base_ci_low": 0.0,
#@        "origin_base_ci_high": 0.0,
#@
#@        "origin_post_period": periods["origin_post_period"],
#@        "origin_post_event_time": 1,
#@        "origin_post_coef": post_origin["coef"],
#@        "origin_post_model_coef": post_origin["model_coef"],
#@        "origin_post_average_discrete_factor": post_origin["average_discrete_factor"],
#@        "origin_post_se": post_origin["se"],
#@        "origin_post_model_se": post_origin["model_se"],
#@        "origin_post_p": post_origin["p"],
#@        "origin_post_ci_low": post_origin["ci_low"],
#@        "origin_post_ci_high": post_origin["ci_high"],
#@        "origin_post_model_ci_low": post_origin["model_ci_low"],
#@        "origin_post_model_ci_high": post_origin["model_ci_high"],
#@
#@        "mentioned_pre_period": periods["mentioned_pre_period"],
#@        "mentioned_pre_event_time": -2,
#@        "mentioned_pre_coef": pre_mentioned["coef"],
#@        "mentioned_pre_model_coef": pre_mentioned["model_coef"],
#@        "mentioned_pre_average_discrete_factor": pre_mentioned["average_discrete_factor"],
#@        "mentioned_pre_se": pre_mentioned["se"],
#@        "mentioned_pre_model_se": pre_mentioned["model_se"],
#@        "mentioned_pre_p": pre_mentioned["p"],
#@        "mentioned_pre_ci_low": pre_mentioned["ci_low"],
#@        "mentioned_pre_ci_high": pre_mentioned["ci_high"],
#@        "mentioned_pre_model_ci_low": pre_mentioned["model_ci_low"],
#@        "mentioned_pre_model_ci_high": pre_mentioned["model_ci_high"],
#@
#@        "mentioned_base_period": periods["mentioned_base_period"],
#@        "mentioned_base_event_time": -1,
#@        "mentioned_base_coef": 0.0,
#@        "mentioned_base_se": 0.0,
#@        "mentioned_base_p": np.nan,
#@        "mentioned_base_ci_low": 0.0,
#@        "mentioned_base_ci_high": 0.0,
#@
#@        "mentioned_post_period": periods["mentioned_post_period"],
#@        "mentioned_post_event_time": 1,
#@        "mentioned_post_coef": post_mentioned["coef"],
#@        "mentioned_post_model_coef": post_mentioned["model_coef"],
#@        "mentioned_post_average_discrete_factor": post_mentioned["average_discrete_factor"],
#@        "mentioned_post_se": post_mentioned["se"],
#@        "mentioned_post_model_se": post_mentioned["model_se"],
#@        "mentioned_post_p": post_mentioned["p"],
#@        "mentioned_post_ci_low": post_mentioned["ci_low"],
#@        "mentioned_post_ci_high": post_mentioned["ci_high"],
#@        "mentioned_post_model_ci_low": post_mentioned["model_ci_low"],
#@        "mentioned_post_model_ci_high": post_mentioned["model_ci_high"],
#@
#@        "estimation_status": "success",
#@        "error_message": "",
#@    }
#@
#@
#@# ============================================================
#@# RUN ONE SPEC - 2022 DID VERSION
#@# ============================================================
#@def run_one_spec_2022(combined_path, yearly_path, asylum_path, ycol, treat_col,
#@                      min_count=30, include_self=False, use_weights=False,
#@                      cluster_mode="pair", treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                      dataset_type="did2022"):
#@
#@    try:
#@        # A. PRE-TREND: 2010-2014 vs 2015-2021
#@        pre_panel = build_pre_panel_2022_from_yearly(
#@            yearly_path=yearly_path,
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@        pre_panel = attach_treatment_two_sides(
#@            panel=pre_panel,
#@            asylum_path=asylum_path,
#@            treat_col=treat_col,
#@            treat_year=2022,
#@            treatment_transform=treatment_transform
#@        )
#@        pre_panel = pre_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
#@        pre_panel, _ = keep_common_pairs(pre_panel, ["2010-2014", "2015-2021"])
#@        pre_panel = build_pretrend_vars_2022(pre_panel)
#@
#@        pre_res, _ = run_panel_ols(
#@            pre_panel,
#@            ["PRE_origin", "PRE_mentioned"],
#@            cluster_mode,
#@            use_weights
#@        )
#@        pre_coef = coef_table_from_res(pre_res)
#@        pre_coef = convert_coef_table_to_average_discrete_effect(
#@            pre_coef,
#@            pre_panel,
#@            {
#@                "PRE_origin": "raw_treat_origin",
#@                "PRE_mentioned": "raw_treat_mentioned",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@
#@        # B. POST-DID: 2015-2021 vs 2022-2024
#@        post_panel = read_panel_from_period_file(
#@            combined_path=combined_path,
#@            sheets=["2015-2021", "2022-2024"],
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@        post_panel, _ = keep_common_pairs(post_panel, ["2015-2021", "2022-2024"])
#@        post_panel = attach_treatment_two_sides(
#@            panel=post_panel,
#@            asylum_path=asylum_path,
#@            treat_col=treat_col,
#@            treat_year=2022,
#@            treatment_transform=treatment_transform
#@        )
#@        post_panel = post_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
#@        post_panel = build_post_did_vars_2022(post_panel)
#@
#@        post_res, _ = run_panel_ols(
#@            post_panel,
#@            ["POST_origin", "POST_mentioned"],
#@            cluster_mode,
#@            use_weights
#@        )
#@        post_coef = coef_table_from_res(post_res)
#@        post_coef = convert_coef_table_to_average_discrete_effect(
#@            post_coef,
#@            post_panel,
#@            {
#@                "POST_origin": "raw_treat_origin",
#@                "POST_mentioned": "raw_treat_mentioned",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@
#@        return make_final_summary_row_common(
#@            dataset_type=dataset_type,
#@            ycol=ycol,
#@            treat_col=treat_col,
#@            pre_coef_df=pre_coef,
#@            post_coef_df=post_coef,
#@            pre_panel=pre_panel,
#@            post_panel=post_panel,
#@            periods={
#@                "origin_pre_period": "2010-2014",
#@                "origin_base_period": "2015-2021",
#@                "origin_post_period": "2022-2024",
#@                "mentioned_pre_period": "2010-2014",
#@                "mentioned_base_period": "2015-2021",
#@                "mentioned_post_period": "2022-2024",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@    except Exception as e:
#@        return make_failed_summary_row_generic(
#@            dataset_type, ycol, treat_col, e, treatment_transform
#@        )
#@
#@
#@# ============================================================
#@# RUN ONE SPEC - 2015 CRISIS VERSION
#@# ============================================================
#@def run_one_spec_2015(combined_path, yearly_path, asylum_path, ycol, treat_col,
#@                      min_count=20, include_self=False, use_weights=False,
#@                      cluster_mode="pair", treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                      dataset_type="crisis2015"):
#@
#@    try:
#@        # A. PRE-TREND: 2010-2012 vs 2013-2014
#@        pre_panel = build_pre_panel_2015_from_yearly(
#@            yearly_path=yearly_path,
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@        pre_panel = attach_dyadic_treat_k_2015(
#@            pre_panel,
#@            asylum_path,
#@            treat_col=treat_col,
#@            treatment_transform=treatment_transform,
#@        )
#@        pre_panel = pre_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
#@        pre_panel, _ = keep_common_pairs(pre_panel, ["2010-2012", "2013-2014"])
#@        pre_panel = build_pretrend_vars_2015(pre_panel)
#@
#@        pre_res, _ = run_panel_ols(
#@            pre_panel,
#@            ["PRE_origin", "PRE_mentioned"],
#@            cluster_mode,
#@            use_weights
#@        )
#@        pre_coef = coef_table_from_res(pre_res)
#@        pre_coef = convert_coef_table_to_average_discrete_effect(
#@            pre_coef,
#@            pre_panel,
#@            {
#@                "PRE_origin": "raw_treat_origin_2015",
#@                "PRE_mentioned": "raw_treat_mentioned_2015",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@
#@        # B. DID POST: 2013-2014 vs 2015-2021
#@        baseline_panel = build_pre_panel_2015_from_yearly(
#@            yearly_path=yearly_path,
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@        baseline_panel = baseline_panel[baseline_panel["period"] == "2013-2014"].copy()
#@
#@        post_period_panel = read_panel_from_period_file(
#@            combined_path=combined_path,
#@            sheets=["2015-2021"],
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@
#@        post_panel = pd.concat([
#@            baseline_panel[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den"]],
#@            post_period_panel[["pair", "origin", "mentioned", "period", "Y", "post_count"]]
#@        ], ignore_index=True)
#@
#@        post_panel, _ = keep_common_pairs(post_panel, ["2013-2014", "2015-2021"])
#@        post_panel = attach_dyadic_treat_k_2015(
#@            post_panel,
#@            asylum_path,
#@            treat_col=treat_col,
#@            treatment_transform=treatment_transform,
#@        )
#@        post_panel = post_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
#@        post_panel = build_post_did_vars_2015(post_panel)
#@
#@        post_res, _ = run_panel_ols(
#@            post_panel,
#@            ["POST_origin", "POST_mentioned"],
#@            cluster_mode,
#@            use_weights
#@        )
#@        post_coef = coef_table_from_res(post_res)
#@        post_coef = convert_coef_table_to_average_discrete_effect(
#@            post_coef,
#@            post_panel,
#@            {
#@                "POST_origin": "raw_treat_origin_2015",
#@                "POST_mentioned": "raw_treat_mentioned_2015",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@
#@        return make_final_summary_row_common(
#@            dataset_type=dataset_type,
#@            ycol=ycol,
#@            treat_col=treat_col,
#@            pre_coef_df=pre_coef,
#@            post_coef_df=post_coef,
#@            pre_panel=pre_panel,
#@            post_panel=post_panel,
#@            periods={
#@                "origin_pre_period": "2010-2012",
#@                "origin_base_period": "2013-2014",
#@                "origin_post_period": "2015-2021",
#@                "mentioned_pre_period": "2010-2012",
#@                "mentioned_base_period": "2013-2014",
#@                "mentioned_post_period": "2015-2021",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@    except Exception as e:
#@        return make_failed_summary_row_generic(
#@            dataset_type, ycol, treat_col, e, treatment_transform
#@        )
#@
#@
#@# ============================================================
#@# RUN ONE SPEC - PLACEBO 2022
#@# ============================================================
#@def run_one_spec_placebo_2022(yearly_path, asylum_path, ycol, treat_col,
#@                              min_count=30, include_self=False, use_weights=False,
#@                              cluster_mode="pair", treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                              dataset_type="placebo2022"):
#@
#@    try:
#@        # A. PRE-TREND: 2010-2012 vs 2013-2014
#@        pre_panel = build_placebo_2022_pre_panel_from_yearly(
#@            yearly_path=yearly_path,
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@        pre_panel = attach_treatment_two_sides(
#@            panel=pre_panel,
#@            asylum_path=asylum_path,
#@            treat_col=treat_col,
#@            treat_year=2022,
#@            treatment_transform=treatment_transform
#@        )
#@        pre_panel = pre_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
#@        pre_panel, _ = keep_common_pairs(pre_panel, ["2010-2012", "2013-2014"])
#@        pre_panel = build_pretrend_vars_placebo_2022(pre_panel)
#@
#@        pre_res, _ = run_panel_ols(
#@            pre_panel,
#@            ["PRE_origin", "PRE_mentioned"],
#@            cluster_mode,
#@            use_weights
#@        )
#@        pre_coef = coef_table_from_res(pre_res)
#@        pre_coef = convert_coef_table_to_average_discrete_effect(
#@            pre_coef,
#@            pre_panel,
#@            {
#@                "PRE_origin": "raw_treat_origin",
#@                "PRE_mentioned": "raw_treat_mentioned",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@
#@        # B. PLACEBO POST: 2013-2014 vs 2015-2021
#@        post_panel = build_placebo_2022_post_panel_from_yearly(
#@            yearly_path=yearly_path,
#@            ycol=ycol,
#@            min_count=min_count,
#@            include_self=include_self
#@        )
#@        post_panel, _ = keep_common_pairs(post_panel, ["2013-2014", "2015-2021"])
#@        post_panel = attach_treatment_two_sides(
#@            panel=post_panel,
#@            asylum_path=asylum_path,
#@            treat_col=treat_col,
#@            treat_year=2022,
#@            treatment_transform=treatment_transform
#@        )
#@        post_panel = post_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
#@        post_panel = build_post_did_vars_placebo_2022(post_panel)
#@
#@        post_res, _ = run_panel_ols(
#@            post_panel,
#@            ["POST_origin", "POST_mentioned"],
#@            cluster_mode,
#@            use_weights
#@        )
#@        post_coef = coef_table_from_res(post_res)
#@        post_coef = convert_coef_table_to_average_discrete_effect(
#@            post_coef,
#@            post_panel,
#@            {
#@                "POST_origin": "raw_treat_origin",
#@                "POST_mentioned": "raw_treat_mentioned",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@
#@        return make_final_summary_row_common(
#@            dataset_type=dataset_type,
#@            ycol=ycol,
#@            treat_col=treat_col,
#@            pre_coef_df=pre_coef,
#@            post_coef_df=post_coef,
#@            pre_panel=pre_panel,
#@            post_panel=post_panel,
#@            periods={
#@                "origin_pre_period": "2010-2012",
#@                "origin_base_period": "2013-2014",
#@                "origin_post_period": "2015-2021",
#@                "mentioned_pre_period": "2010-2012",
#@                "mentioned_base_period": "2013-2014",
#@                "mentioned_post_period": "2015-2021",
#@            },
#@            treatment_transform=treatment_transform,
#@        )
#@    except Exception as e:
#@        return make_failed_summary_row_generic(
#@            dataset_type, ycol, treat_col, e, treatment_transform
#@        )
#@
#@
#@# ============================================================
#@# RUN ONE SPEC - PLACEBO 2015 WITH FALLBACK
#@# ============================================================
#@def run_one_spec_placebo_2015(yearly_path, asylum_path, ycol, treat_col,
#@                              min_count=20, include_self=False, use_weights=False,
#@                              cluster_mode="pair", treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                              dataset_type="placebo2015"):
#@
#@    errors = []
#@
#@    for spec in PLACEBO_2015_SPECS:
#@        try:
#@            spec_name = spec["name"]
#@
#@            # A. PRE-TREND
#@            pre_panel = build_yearly_period_panel_from_mapping(
#@                yearly_path=yearly_path,
#@                target_years=spec["pre_target_years"],
#@                period_map_func=placebo2015_pre_mapper_factory(spec_name),
#@                ycol=ycol,
#@                min_count=min_count,
#@                include_self=include_self
#@            )
#@            pre_panel = attach_dyadic_treat_k_2015(
#@                pre_panel,
#@                asylum_path,
#@                treat_col=treat_col,
#@                treatment_transform=treatment_transform,
#@            )
#@            pre_panel = pre_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
#@            pre_panel, _ = keep_common_pairs(pre_panel, spec["pre_periods"])
#@            pre_panel = build_pretrend_vars_placebo_2015(
#@                pre_panel,
#@                pre_period=spec["pre_periods"][0],
#@                base_period=spec["pre_periods"][1]
#@            )
#@
#@            pre_res, _ = run_panel_ols(
#@                pre_panel,
#@                ["PRE_origin", "PRE_mentioned"],
#@                cluster_mode,
#@                use_weights
#@            )
#@            pre_coef = coef_table_from_res(pre_res)
#@            pre_coef = convert_coef_table_to_average_discrete_effect(
#@                pre_coef,
#@                pre_panel,
#@                {
#@                    "PRE_origin": "raw_treat_origin_2015",
#@                    "PRE_mentioned": "raw_treat_mentioned_2015",
#@                },
#@                treatment_transform=treatment_transform,
#@            )
#@
#@            # B. PLACEBO POST
#@            post_panel = build_yearly_period_panel_from_mapping(
#@                yearly_path=yearly_path,
#@                target_years=spec["post_target_years"],
#@                period_map_func=placebo2015_post_mapper_factory(spec_name),
#@                ycol=ycol,
#@                min_count=min_count,
#@                include_self=include_self
#@            )
#@            post_panel = attach_dyadic_treat_k_2015(
#@                post_panel,
#@                asylum_path,
#@                treat_col=treat_col,
#@                treatment_transform=treatment_transform,
#@            )
#@            post_panel = post_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
#@            post_panel, _ = keep_common_pairs(post_panel, spec["post_periods"])
#@            post_panel = build_post_did_vars_placebo_2015(
#@                post_panel,
#@                base_period=spec["post_periods"][0],
#@                post_period=spec["post_periods"][1]
#@            )
#@
#@            post_res, _ = run_panel_ols(
#@                post_panel,
#@                ["POST_origin", "POST_mentioned"],
#@                cluster_mode,
#@                use_weights
#@            )
#@            post_coef = coef_table_from_res(post_res)
#@            post_coef = convert_coef_table_to_average_discrete_effect(
#@                post_coef,
#@                post_panel,
#@                {
#@                    "POST_origin": "raw_treat_origin_2015",
#@                    "POST_mentioned": "raw_treat_mentioned_2015",
#@                },
#@                treatment_transform=treatment_transform,
#@            )
#@
#@            out = make_final_summary_row_common(
#@                dataset_type=dataset_type,
#@                ycol=ycol,
#@                treat_col=treat_col,
#@                pre_coef_df=pre_coef,
#@                post_coef_df=post_coef,
#@                pre_panel=pre_panel,
#@                post_panel=post_panel,
#@                periods=spec["period_labels"],
#@                treatment_transform=treatment_transform,
#@            )
#@            out["placebo2015_spec_used"] = spec_name
#@            return out
#@
#@        except Exception as e:
#@            errors.append(f"{spec['name']}: {str(e)}")
#@
#@    return make_failed_summary_row_generic(
#@        dataset_type, ycol, treat_col,
#@        " | ".join(errors), treatment_transform
#@    )
#@
#@
#@# ============================================================
#@# DRAW FIGURES FROM TWO MASTER CSV FILES
#@# ============================================================
#@def format_period_labels_type_2022(periods):
#@    mapping = {
#@        "2010-2014": "2010–2014\nPre-period",
#@        "2015-2021": "2015–2021\nReference",
#@        "2022-2024": "2022–2024\nPost-period",
#@    }
#@    return [mapping.get(p, str(p)) for p in periods]
#@
#@
#@def format_period_labels_type_2015(periods):
#@    mapping = {
#@        "2010-2012": "2010–2012\nPre-period",
#@        "2013-2014": "2013–2014\nReference",
#@        "2015-2021": "2015–2021\nPost-period",
#@    }
#@    return [mapping.get(p, str(p)) for p in periods]
#@
#@
#@def format_period_labels_type_placebo2022(periods):
#@    mapping = {
#@        "2010-2012": "2010–2012\nPre-period",
#@        "2013-2014": "2013–2014\nReference",
#@        "2015-2021": "2015–2021\nPlacebo post",
#@    }
#@    return [mapping.get(p, str(p)) for p in periods]
#@
#@
#@def format_period_labels_type_placebo2015(periods):
#@    mapping = {
#@        "2010-2011": "2010–2011\nPre-period",
#@        "2012-2013": "2012–2013\nReference",
#@        "2014": "2014\nPlacebo post",
#@        "2010-2012": "2010–2012\nPre-period",
#@        "2013": "2013\nReference",
#@        "2013-2014": "2013–2014\nPlacebo post",
#@    }
#@    return [mapping.get(p, str(p)) for p in periods]
#@
#@
#@def get_series_common(row, side):
#@    x = unified_x()
#@
#@    if side == "origin":
#@        periods = [row["origin_pre_period"], row["origin_base_period"], row["origin_post_period"]]
#@        y = [
#@            row["origin_pre_coef"],
#@            row["origin_base_coef"],
#@            row["origin_post_coef"]
#@        ]
#@        ci_low = [
#@            row["origin_pre_ci_low"],
#@            row["origin_base_ci_low"],
#@            row["origin_post_ci_low"]
#@        ]
#@        ci_high = [
#@            row["origin_pre_ci_high"],
#@            row["origin_base_ci_high"],
#@            row["origin_post_ci_high"]
#@        ]
#@    else:
#@        periods = [row["mentioned_pre_period"], row["mentioned_base_period"], row["mentioned_post_period"]]
#@        y = [
#@            row["mentioned_pre_coef"],
#@            row["mentioned_base_coef"],
#@            row["mentioned_post_coef"]
#@        ]
#@        ci_low = [
#@            row["mentioned_pre_ci_low"],
#@            row["mentioned_base_ci_low"],
#@            row["mentioned_post_ci_low"]
#@        ]
#@        ci_high = [
#@            row["mentioned_pre_ci_high"],
#@            row["mentioned_base_ci_high"],
#@            row["mentioned_post_ci_high"]
#@        ]
#@
#@    yerr_lower = [y[i] - ci_low[i] if pd.notna(y[i]) and pd.notna(ci_low[i]) else np.nan for i in range(3)]
#@    yerr_upper = [ci_high[i] - y[i] if pd.notna(y[i]) and pd.notna(ci_high[i]) else np.nan for i in range(3)]
#@
#@    return x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high
#@
#@
#@def build_shared_special_ylim(df1, df2):
#@    all_ci_low = []
#@    all_ci_high = []
#@
#@    sub1 = df1[
#@        (df1["outcome"].astype(str) == SPECIAL_OUTCOME)
#@        & (df1["treatment"].astype(str).isin(SPECIAL_TREATMENTS))
#@    ]
#@
#@    if len(sub1) > 0:
#@        row1 = sub1.iloc[0]
#@        _, _, _, _, _, ci_low1, ci_high1 = get_series_common(row1, SPECIAL_SIDE)
#@        all_ci_low.extend(ci_low1)
#@        all_ci_high.extend(ci_high1)
#@
#@    sub2 = df2[
#@        (df2["outcome"].astype(str) == SPECIAL_OUTCOME)
#@        & (df2["treatment"].astype(str).isin(SPECIAL_TREATMENTS))
#@    ]
#@
#@    if len(sub2) > 0:
#@        row2 = sub2.iloc[0]
#@        _, _, _, _, _, ci_low2, ci_high2 = get_series_common(row2, SPECIAL_SIDE)
#@        all_ci_low.extend(ci_low2)
#@        all_ci_high.extend(ci_high2)
#@
#@    if len(all_ci_low) == 0 or len(all_ci_high) == 0:
#@        return None
#@
#@    return nice_ylim_raw(all_ci_low, all_ci_high)
#@
#@
#@def draw_single_line_plot(row, side, dataset_type, prefix, output_dir, special_shared_ylim=None):
#@    outcome_value = row["outcome"]
#@    color = get_line_color(outcome_value)
#@
#@    display_prefix = display_prefix_for_dataset_type(prefix)
#@
#@    if row.get("estimation_status", "success") == "failed":
#@        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
#@        ax.axis("off")
#@        msg = str(row.get("error_message", "Estimation failed"))
#@        ax.text(0.02, 0.95, msg, va="top", ha="left", fontsize=10, wrap=True)
#@
#@        fname = make_figure_filename(
#@            prefix=display_prefix,
#@            outcome=row["outcome"],
#@            treatment=row["treatment"],
#@            side=side
#@        )
#@
#@        plt.tight_layout()
#@        plt.savefig(
#@            os.path.join(output_dir, fname),
#@            dpi=600,
#@            bbox_inches="tight",
#@            facecolor="white"
#@        )
#@        if SAVE_PDF:
#@            pdf_name = os.path.splitext(fname)[0] + ".pdf"
#@            plt.savefig(
#@                os.path.join(output_dir, pdf_name),
#@                bbox_inches="tight",
#@                facecolor="white"
#@            )
#@        plt.close()
#@        return
#@
#@    x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high = get_series_common(row, side)
#@
#@    if dataset_type == "did2022":
#@        labels = format_period_labels_type_2022(periods)
#@    elif dataset_type == "crisis2015":
#@        labels = format_period_labels_type_2015(periods)
#@    elif dataset_type == "placebo2022":
#@        labels = format_period_labels_type_placebo2022(periods)
#@    elif dataset_type == "placebo2015":
#@        labels = format_period_labels_type_placebo2015(periods)
#@    else:
#@        labels = [str(p) for p in periods]
#@
#@    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
#@
#@    # Plot only estimated coefficients.  The middle period is the omitted
#@    # reference category fixed at zero and therefore has no estimated marker.
#@    for i in (0, 2):
#@        draw_coef_point(
#@            ax=ax,
#@            x_value=x[i],
#@            y_value=y[i],
#@            yerr_lower=yerr_lower[i],
#@            yerr_upper=yerr_upper[i],
#@            ci_low=ci_low[i],
#@            ci_high=ci_high[i],
#@            color=color,
#@            zorder=4
#@        )
#@
#@    is_special = use_special_tick_labels(row, side, display_prefix)
#@
#@    if is_special and special_shared_ylim is not None:
#@        ymin, ymax = special_shared_ylim
#@    else:
#@        ymin, ymax = nice_ylim_raw(ci_low, ci_high)
#@
#@    ax.set_ylim(ymin, ymax)
#@    style_axis(ax)
#@
#@    if is_special and RELABEL_SPECIAL_YTICKS:
#@        relabel_top_bottom_yticks(
#@            ax,
#@            bottom_label=SPECIAL_BOTTOM_LABEL,
#@            top_label=SPECIAL_TOP_LABEL
#@        )
#@
#@    ax.set_xticks(x)
#@    ax.set_xticklabels(labels)
#@
#@    if SHOW_YLABEL:
#@        if str(outcome_value) == "mean_sentiment_score":
#@            outcome_axis_text = "sentiment"
#@        elif str(outcome_value) == "mean_hate_speech_score":
#@            outcome_axis_text = "hate speech"
#@        else:
#@            outcome_axis_text = "outcome"
#@
#@        ax.set_ylabel(
#@            f"Average change in {outcome_axis_text}\n"
#@            "per +1 applicant per 1,000 residents",
#@            fontsize=LABEL_SIZE,
#@        )
#@
#@    if SHOW_POST_LABEL and pd.notna(y[2]):
#@        label = format_coef_label(y[2], ci_low[2], ci_high[2])
#@
#@        ax.annotate(
#@            label,
#@            xy=(x[2], y[2]),
#@            xytext=(9, 0),
#@            textcoords="offset points",
#@            ha="left",
#@            va="center",
#@            fontsize=ANNOT_SIZE,
#@            color=color,
#@            fontweight="bold",
#@            zorder=6
#@        )
#@
#@    fname = make_figure_filename(
#@        prefix=display_prefix,
#@        outcome=row["outcome"],
#@        treatment=row["treatment"],
#@        side=side
#@    )
#@
#@    plt.tight_layout()
#@
#@    plt.savefig(
#@        os.path.join(output_dir, fname),
#@        dpi=600,
#@        bbox_inches="tight",
#@        facecolor="white"
#@    )
#@
#@    if SAVE_PDF:
#@        pdf_name = fname.replace(".png", ".pdf")
#@        plt.savefig(
#@            os.path.join(output_dir, pdf_name),
#@            bbox_inches="tight",
#@            facecolor="white"
#@        )
#@
#@    plt.close()
#@
#@
#@def draw_all_figures(master_2022did_csv, master_2015_csv, figure_dir, dataset_type_2022, dataset_type_2015):
#@    safe_mkdir(figure_dir)
#@
#@    df1 = pd.read_csv(master_2022did_csv)
#@    df2 = pd.read_csv(master_2015_csv)
#@
#@    special_shared_ylim = build_shared_special_ylim(df1, df2)
#@
#@    for _, row in df1.iterrows():
#@        draw_single_line_plot(
#@            row,
#@            "origin",
#@            dataset_type_2022,
#@            dataset_type_2022,
#@            figure_dir,
#@            special_shared_ylim
#@        )
#@        draw_single_line_plot(
#@            row,
#@            "mentioned",
#@            dataset_type_2022,
#@            dataset_type_2022,
#@            figure_dir,
#@            special_shared_ylim
#@        )
#@
#@    for _, row in df2.iterrows():
#@        draw_single_line_plot(
#@            row,
#@            "origin",
#@            dataset_type_2015,
#@            dataset_type_2015,
#@            figure_dir,
#@            special_shared_ylim
#@        )
#@        draw_single_line_plot(
#@            row,
#@            "mentioned",
#@            dataset_type_2015,
#@            dataset_type_2015,
#@            figure_dir,
#@            special_shared_ylim
#@        )
#@
#@
#@# ============================================================
#@# MAIN RUNNERS FOR DIFFERENT CHECKS
#@# ============================================================
#@def run_2022_version(master_out_dir, dataset_type, use_weights=False, cluster_mode="pair",
#@                     treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                     did2022_min_count=None):
#@    master_summary = []
#@
#@    if did2022_min_count is None:
#@        did2022_min_count = DID2022_MIN_COUNT
#@
#@    for outcome in DID2022_OUTCOME_SPECS:
#@        ycol = outcome["ycol"]
#@
#@        for treat in DID2022_TREAT_SPECS:
#@            treat_col = treat["treat_col"]
#@            print(f"[{dataset_type}] Running 2022: Y={ycol}, Treat={treat_col}, min_count={did2022_min_count}")
#@
#@            if dataset_type == "placebo2022":
#@                summary_row = run_one_spec_placebo_2022(
#@                    yearly_path=yearly_path,
#@                    asylum_path=asylum_path,
#@                    ycol=ycol,
#@                    treat_col=treat_col,
#@                    min_count=did2022_min_count,
#@                    include_self=INCLUDE_SELF,
#@                    use_weights=use_weights,
#@                    cluster_mode=cluster_mode,
#@                    treatment_transform=treatment_transform,
#@                    dataset_type=dataset_type
#@                )
#@            else:
#@                summary_row = run_one_spec_2022(
#@                    combined_path=combined_path,
#@                    yearly_path=yearly_path,
#@                    asylum_path=asylum_path,
#@                    ycol=ycol,
#@                    treat_col=treat_col,
#@                    min_count=did2022_min_count,
#@                    include_self=INCLUDE_SELF,
#@                    use_weights=use_weights,
#@                    cluster_mode=cluster_mode,
#@                    treatment_transform=treatment_transform,
#@                    dataset_type=dataset_type
#@                )
#@            master_summary.append(summary_row)
#@
#@    out_csv = os.path.join(master_out_dir, "results_2022crisis.csv")
#@    pd.DataFrame(master_summary).to_csv(out_csv, index=False, encoding="utf-8-sig")
#@    return out_csv
#@
#@
#@def run_2015_version(master_out_dir, dataset_type, use_weights=False, cluster_mode="pair",
#@                     treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                     crisis2015_min_count=None):
#@    master_summary = []
#@
#@    if crisis2015_min_count is None:
#@        crisis2015_min_count = CRISIS2015_MIN_COUNT
#@
#@    for outcome in CRISIS2015_OUTCOME_SPECS:
#@        ycol = outcome["ycol"]
#@
#@        for treat in CRISIS2015_TREAT_SPECS:
#@            treat_col = treat["treat_col"]
#@            print(f"[{dataset_type}] Running 2015: Y={ycol}, Treat={treat_col}, min_count={crisis2015_min_count}")
#@
#@            if dataset_type == "placebo2015":
#@                summary_row = run_one_spec_placebo_2015(
#@                    yearly_path=yearly_path,
#@                    asylum_path=asylum_path,
#@                    ycol=ycol,
#@                    treat_col=treat_col,
#@                    min_count=crisis2015_min_count,
#@                    include_self=INCLUDE_SELF,
#@                    use_weights=use_weights,
#@                    cluster_mode=cluster_mode,
#@                    treatment_transform=treatment_transform,
#@                    dataset_type=dataset_type
#@                )
#@            else:
#@                summary_row = run_one_spec_2015(
#@                    combined_path=combined_path,
#@                    yearly_path=yearly_path,
#@                    asylum_path=asylum_path,
#@                    ycol=ycol,
#@                    treat_col=treat_col,
#@                    min_count=crisis2015_min_count,
#@                    include_self=INCLUDE_SELF,
#@                    use_weights=use_weights,
#@                    cluster_mode=cluster_mode,
#@                    treatment_transform=treatment_transform,
#@                    dataset_type=dataset_type
#@                )
#@            master_summary.append(summary_row)
#@
#@    out_csv = os.path.join(master_out_dir, "results_2015crisis.csv")
#@    pd.DataFrame(master_summary).to_csv(out_csv, index=False, encoding="utf-8-sig")
#@    return out_csv
#@
#@
#@def run_one_check(output_folder, dataset_type_2022, dataset_type_2015,
#@                  use_weights=False, cluster_mode="pair",
#@                  treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@                  did2022_min_count=None, crisis2015_min_count=None):
#@    safe_mkdir(output_folder)
#@
#@    figure_dir = os.path.join(output_folder, "figures")
#@    safe_mkdir(figure_dir)
#@
#@    master_2022did_csv = run_2022_version(
#@        master_out_dir=output_folder,
#@        dataset_type=dataset_type_2022,
#@        use_weights=use_weights,
#@        cluster_mode=cluster_mode,
#@        treatment_transform=treatment_transform,
#@        did2022_min_count=did2022_min_count
#@    )
#@    master_2015_csv = run_2015_version(
#@        master_out_dir=output_folder,
#@        dataset_type=dataset_type_2015,
#@        use_weights=use_weights,
#@        cluster_mode=cluster_mode,
#@        treatment_transform=treatment_transform,
#@        crisis2015_min_count=crisis2015_min_count
#@    )
#@
#@    draw_all_figures(
#@        master_2022did_csv=master_2022did_csv,
#@        master_2015_csv=master_2015_csv,
#@        figure_dir=figure_dir,
#@        dataset_type_2022=dataset_type_2022,
#@        dataset_type_2015=dataset_type_2015
#@    )
#@
#@
#@# ============================================================
#@# ROBUSTNESS CHECKS
#@# ============================================================
#@def run_weighted_check():
#@    out_dir = os.path.join(OUT_ROOT, "weighted")
#@    run_one_check(
#@        output_folder=out_dir,
#@        dataset_type_2022="did2022",
#@        dataset_type_2015="crisis2015",
#@        use_weights=True,
#@        cluster_mode="pair",
#@        treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@        did2022_min_count=DID2022_MIN_COUNT,
#@        crisis2015_min_count=CRISIS2015_MIN_COUNT
#@    )
#@
#@
#@def run_cluster_robustness_check():
#@    root_dir = os.path.join(OUT_ROOT, "cluster_robustness")
#@    safe_mkdir(root_dir)
#@
#@    for cm in ["pair", "origin_mentioned", "origin", "mentioned"]:
#@        out_dir = os.path.join(root_dir, cm)
#@        run_one_check(
#@            output_folder=out_dir,
#@            dataset_type_2022="did2022",
#@            dataset_type_2015="crisis2015",
#@            use_weights=False,
#@            cluster_mode=cm,
#@            treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@            did2022_min_count=DID2022_MIN_COUNT,
#@            crisis2015_min_count=CRISIS2015_MIN_COUNT
#@        )
#@
#@
#@def run_raw_treatment_check():
#@    """Sensitivity check using the original linear treatment intensity."""
#@    out_dir = os.path.join(OUT_ROOT, "raw_treatment")
#@    run_one_check(
#@        output_folder=out_dir,
#@        dataset_type_2022="did2022",
#@        dataset_type_2015="crisis2015",
#@        use_weights=False,
#@        cluster_mode="pair",
#@        treatment_transform="raw",
#@        did2022_min_count=DID2022_MIN_COUNT,
#@        crisis2015_min_count=CRISIS2015_MIN_COUNT
#@    )
#@
#@
#@def run_placebo_check():
#@    out_dir = os.path.join(OUT_ROOT, "placebo_test")
#@    run_one_check(
#@        output_folder=out_dir,
#@        dataset_type_2022="placebo2022",
#@        dataset_type_2015="placebo2015",
#@        use_weights=False,
#@        cluster_mode="pair",
#@        treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@        did2022_min_count=DID2022_MIN_COUNT,
#@        crisis2015_min_count=CRISIS2015_MIN_COUNT
#@    )
#@
#@
#@def run_threshold_plus20_check():
#@    out_dir = os.path.join(OUT_ROOT, "threshold_plus20")
#@    run_one_check(
#@        output_folder=out_dir,
#@        dataset_type_2022="did2022",
#@        dataset_type_2015="crisis2015",
#@        use_weights=False,
#@        cluster_mode="pair",
#@        treatment_transform=MAIN_TREATMENT_TRANSFORM,
#@        did2022_min_count=THRESHOLD_PLUS20_MIN_COUNT_2022,
#@        crisis2015_min_count=THRESHOLD_PLUS20_MIN_COUNT_2015
#@    )
#@
#@
#@# ============================================================
#@# SIGNIFICANCE SUMMARY EXCEL
#@# ============================================================
#@def read_summary_csv_if_exists(path, check_name, crisis_label, cluster_mode=""):
#@    if not os.path.exists(path):
#@        return pd.DataFrame()
#@
#@    df = pd.read_csv(path).copy()
#@    if df.empty:
#@        return df
#@
#@    df["check_name"] = check_name
#@    df["crisis_label"] = crisis_label
#@    df["cluster_mode"] = cluster_mode
#@
#@    needed_cols = [
#@        "dataset_type", "outcome", "treatment", "treatment_transform",
#@        "model_treatment_scale", "reported_effect_scale",
#@        "origin_parallel", "mentioned_parallel",
#@        "origin_pre_coef", "origin_pre_model_coef",
#@        "origin_pre_average_discrete_factor", "origin_pre_p",
#@        "mentioned_pre_coef", "mentioned_pre_model_coef",
#@        "mentioned_pre_average_discrete_factor", "mentioned_pre_p",
#@        "origin_post_coef", "origin_post_model_coef",
#@        "origin_post_average_discrete_factor", "origin_post_p",
#@        "mentioned_post_coef", "mentioned_post_model_coef",
#@        "mentioned_post_average_discrete_factor", "mentioned_post_p",
#@        "estimation_status"
#@    ]
#@    for c in needed_cols:
#@        if c not in df.columns:
#@            df[c] = np.nan
#@
#@    df["origin_parallel_pass"] = df["origin_parallel"].astype(str).str.lower().eq("pass")
#@    df["mentioned_parallel_pass"] = df["mentioned_parallel"].astype(str).str.lower().eq("pass")
#@
#@    df["origin_did_significant"] = pd.to_numeric(df["origin_post_p"], errors="coerce") < 0.05
#@    df["mentioned_did_significant"] = pd.to_numeric(df["mentioned_post_p"], errors="coerce") < 0.05
#@
#@    df["origin_model_significant"] = (
#@        df["estimation_status"].astype(str).str.lower().eq("success")
#@        & df["origin_parallel_pass"]
#@        & df["origin_did_significant"]
#@    )
#@
#@    df["mentioned_model_significant"] = (
#@        df["estimation_status"].astype(str).str.lower().eq("success")
#@        & df["mentioned_parallel_pass"]
#@        & df["mentioned_did_significant"]
#@    )
#@
#@    df["any_side_significant"] = (
#@        df["origin_model_significant"] | df["mentioned_model_significant"]
#@    )
#@
#@    df["both_sides_significant"] = (
#@        df["origin_model_significant"] & df["mentioned_model_significant"]
#@    )
#@
#@    df["origin_result"] = np.where(
#@        df["origin_model_significant"],
#@        "significant",
#@        "not significant"
#@    )
#@    df["mentioned_result"] = np.where(
#@        df["mentioned_model_significant"],
#@        "significant",
#@        "not significant"
#@    )
#@
#@    return df
#@
#@
#@def build_significance_master_table():
#@    all_parts = []
#@
#@    # weighted
#@    weighted_dir = os.path.join(OUT_ROOT, "weighted")
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(weighted_dir, "results_2022crisis.csv"),
#@            check_name="weighted",
#@            crisis_label="2022"
#@        )
#@    )
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(weighted_dir, "results_2015crisis.csv"),
#@            check_name="weighted",
#@            crisis_label="2015"
#@        )
#@    )
#@
#@    # raw linear treatment sensitivity
#@    raw_dir = os.path.join(OUT_ROOT, "raw_treatment")
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(raw_dir, "results_2022crisis.csv"),
#@            check_name="raw_treatment",
#@            crisis_label="2022"
#@        )
#@    )
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(raw_dir, "results_2015crisis.csv"),
#@            check_name="raw_treatment",
#@            crisis_label="2015"
#@        )
#@    )
#@
#@    # placebo
#@    placebo_dir = os.path.join(OUT_ROOT, "placebo_test")
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(placebo_dir, "results_2022crisis.csv"),
#@            check_name="placebo_test",
#@            crisis_label="2022"
#@        )
#@    )
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(placebo_dir, "results_2015crisis.csv"),
#@            check_name="placebo_test",
#@            crisis_label="2015"
#@        )
#@    )
#@
#@    # threshold +20
#@    plus20_dir = os.path.join(OUT_ROOT, "threshold_plus20")
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(plus20_dir, "results_2022crisis.csv"),
#@            check_name="threshold_plus20",
#@            crisis_label="2022"
#@        )
#@    )
#@    all_parts.append(
#@        read_summary_csv_if_exists(
#@            os.path.join(plus20_dir, "results_2015crisis.csv"),
#@            check_name="threshold_plus20",
#@            crisis_label="2015"
#@        )
#@    )
#@
#@    # cluster robustness
#@    cluster_root = os.path.join(OUT_ROOT, "cluster_robustness")
#@    for cm in ["pair", "origin_mentioned", "origin", "mentioned"]:
#@        cm_dir = os.path.join(cluster_root, cm)
#@
#@        all_parts.append(
#@            read_summary_csv_if_exists(
#@                os.path.join(cm_dir, "results_2022crisis.csv"),
#@                check_name="cluster_robustness",
#@                crisis_label="2022",
#@                cluster_mode=cm
#@            )
#@        )
#@        all_parts.append(
#@            read_summary_csv_if_exists(
#@                os.path.join(cm_dir, "results_2015crisis.csv"),
#@                check_name="cluster_robustness",
#@                crisis_label="2015",
#@                cluster_mode=cm
#@            )
#@        )
#@
#@    all_parts = [x for x in all_parts if x is not None and not x.empty]
#@    if not all_parts:
#@        return pd.DataFrame()
#@
#@    df_all = pd.concat(all_parts, ignore_index=True)
#@
#@    preferred_cols = [
#@        "check_name",
#@        "cluster_mode",
#@        "crisis_label",
#@        "dataset_type",
#@        "outcome",
#@        "treatment",
#@        "treatment_transform",
#@        "model_treatment_scale",
#@        "reported_effect_scale",
#@        "estimation_status",
#@
#@        "origin_parallel",
#@        "origin_parallel_pass",
#@        "origin_pre_coef",
#@        "origin_pre_model_coef",
#@        "origin_pre_average_discrete_factor",
#@        "origin_pre_p",
#@        "origin_post_coef",
#@        "origin_post_model_coef",
#@        "origin_post_average_discrete_factor",
#@        "origin_post_p",
#@        "origin_did_significant",
#@        "origin_model_significant",
#@        "origin_result",
#@
#@        "mentioned_parallel",
#@        "mentioned_parallel_pass",
#@        "mentioned_pre_coef",
#@        "mentioned_pre_model_coef",
#@        "mentioned_pre_average_discrete_factor",
#@        "mentioned_pre_p",
#@        "mentioned_post_coef",
#@        "mentioned_post_model_coef",
#@        "mentioned_post_average_discrete_factor",
#@        "mentioned_post_p",
#@        "mentioned_did_significant",
#@        "mentioned_model_significant",
#@        "mentioned_result",
#@
#@        "any_side_significant",
#@        "both_sides_significant",
#@
#@        "n_pairs_pre_total",
#@        "n_obs_pre",
#@        "n_pairs_post_common",
#@        "n_obs_post",
#@
#@        "error_message"
#@    ]
#@
#@    existing_cols = [c for c in preferred_cols if c in df_all.columns]
#@    other_cols = [c for c in df_all.columns if c not in existing_cols]
#@    df_all = df_all[existing_cols + other_cols]
#@
#@    return df_all
#@
#@
#@def build_significance_overview(df_all):
#@    if df_all.empty:
#@        return pd.DataFrame()
#@
#@    overview = df_all.copy()
#@
#@    overview["overall_result"] = np.select(
#@        [
#@            overview["both_sides_significant"],
#@            overview["any_side_significant"]
#@        ],
#@        [
#@            "both sides significant",
#@            "one side significant"
#@        ],
#@        default="not significant"
#@    )
#@
#@    keep_cols = [
#@        "check_name",
#@        "cluster_mode",
#@        "crisis_label",
#@        "dataset_type",
#@        "outcome",
#@        "treatment",
#@        "treatment_transform",
#@        "model_treatment_scale",
#@        "reported_effect_scale",
#@        "overall_result",
#@        "origin_result",
#@        "mentioned_result",
#@        "origin_post_coef",
#@        "origin_post_model_coef",
#@        "origin_post_average_discrete_factor",
#@        "origin_post_p",
#@        "mentioned_post_coef",
#@        "mentioned_post_model_coef",
#@        "mentioned_post_average_discrete_factor",
#@        "mentioned_post_p",
#@        "origin_parallel",
#@        "origin_pre_coef",
#@        "origin_pre_model_coef",
#@        "origin_pre_p",
#@        "mentioned_parallel",
#@        "mentioned_pre_coef",
#@        "mentioned_pre_model_coef",
#@        "mentioned_pre_p",
#@        "estimation_status",
#@        "error_message"
#@    ]
#@
#@    keep_cols = [c for c in keep_cols if c in overview.columns]
#@    return overview[keep_cols].copy()
#@
#@
#@def build_significant_only_table(df_all):
#@    if df_all.empty:
#@        return pd.DataFrame()
#@    return df_all[df_all["any_side_significant"]].copy()
#@
#@
#@def build_concise_side_table(df_all):
#@    if df_all.empty:
#@        return pd.DataFrame()
#@
#@    parts = []
#@
#@    # ---------------------------
#@    # origin side
#@    # ---------------------------
#@    origin_cols = [
#@        "check_name",
#@        "cluster_mode",
#@        "crisis_label",
#@        "outcome",
#@        "treatment",
#@        "treatment_transform",
#@        "model_treatment_scale",
#@        "reported_effect_scale",
#@        "origin_parallel",
#@        "origin_parallel_pass",
#@        "origin_pre_coef",
#@        "origin_pre_model_coef",
#@        "origin_pre_average_discrete_factor",
#@        "origin_pre_p",
#@        "origin_pre_ci_low",
#@        "origin_pre_ci_high",
#@        "origin_post_coef",
#@        "origin_post_model_coef",
#@        "origin_post_average_discrete_factor",
#@        "origin_post_p",
#@        "origin_post_ci_low",
#@        "origin_post_ci_high",
#@        "origin_did_significant",
#@        "origin_model_significant",
#@        "estimation_status",
#@        "error_message"
#@    ]
#@    df_o = df_all[origin_cols].copy()
#@    df_o["side"] = "origin"
#@    df_o = df_o.rename(columns={
#@        "origin_parallel": "parallel_result",
#@        "origin_parallel_pass": "parallel_pass",
#@        "origin_pre_coef": "parallel_coef",
#@        "origin_pre_model_coef": "parallel_model_coef",
#@        "origin_pre_average_discrete_factor": "parallel_average_discrete_factor",
#@        "origin_pre_p": "parallel_p",
#@        "origin_pre_ci_low": "parallel_ci_low",
#@        "origin_pre_ci_high": "parallel_ci_high",
#@        "origin_post_coef": "did_coef",
#@        "origin_post_model_coef": "did_model_coef",
#@        "origin_post_average_discrete_factor": "did_average_discrete_factor",
#@        "origin_post_p": "did_p",
#@        "origin_post_ci_low": "did_ci_low",
#@        "origin_post_ci_high": "did_ci_high",
#@        "origin_did_significant": "did_significant",
#@        "origin_model_significant": "final_significant"
#@    })
#@    parts.append(df_o)
#@
#@    # ---------------------------
#@    # mentioned side
#@    # ---------------------------
#@    mentioned_cols = [
#@        "check_name",
#@        "cluster_mode",
#@        "crisis_label",
#@        "outcome",
#@        "treatment",
#@        "treatment_transform",
#@        "model_treatment_scale",
#@        "reported_effect_scale",
#@        "mentioned_parallel",
#@        "mentioned_parallel_pass",
#@        "mentioned_pre_coef",
#@        "mentioned_pre_model_coef",
#@        "mentioned_pre_average_discrete_factor",
#@        "mentioned_pre_p",
#@        "mentioned_pre_ci_low",
#@        "mentioned_pre_ci_high",
#@        "mentioned_post_coef",
#@        "mentioned_post_model_coef",
#@        "mentioned_post_average_discrete_factor",
#@        "mentioned_post_p",
#@        "mentioned_post_ci_low",
#@        "mentioned_post_ci_high",
#@        "mentioned_did_significant",
#@        "mentioned_model_significant",
#@        "estimation_status",
#@        "error_message"
#@    ]
#@    df_m = df_all[mentioned_cols].copy()
#@    df_m["side"] = "mentioned"
#@    df_m = df_m.rename(columns={
#@        "mentioned_parallel": "parallel_result",
#@        "mentioned_parallel_pass": "parallel_pass",
#@        "mentioned_pre_coef": "parallel_coef",
#@        "mentioned_pre_model_coef": "parallel_model_coef",
#@        "mentioned_pre_average_discrete_factor": "parallel_average_discrete_factor",
#@        "mentioned_pre_p": "parallel_p",
#@        "mentioned_pre_ci_low": "parallel_ci_low",
#@        "mentioned_pre_ci_high": "parallel_ci_high",
#@        "mentioned_post_coef": "did_coef",
#@        "mentioned_post_model_coef": "did_model_coef",
#@        "mentioned_post_average_discrete_factor": "did_average_discrete_factor",
#@        "mentioned_post_p": "did_p",
#@        "mentioned_post_ci_low": "did_ci_low",
#@        "mentioned_post_ci_high": "did_ci_high",
#@        "mentioned_did_significant": "did_significant",
#@        "mentioned_model_significant": "final_significant"
#@    })
#@    parts.append(df_m)
#@
#@    out = pd.concat(parts, ignore_index=True)
#@
#@    out["outcome_label"] = out["outcome"].map({
#@        "mean_sentiment_score": "sentiment",
#@        "mean_hate_speech_score": "hate"
#@    }).fillna(out["outcome"])
#@
#@    out["crisis_label"] = out["crisis_label"].astype(str)
#@    out["check_name"] = out["check_name"].astype(str)
#@    out["cluster_mode"] = out["cluster_mode"].fillna("").astype(str)
#@
#@    preferred = [
#@        "check_name",
#@        "cluster_mode",
#@        "crisis_label",
#@        "side",
#@        "outcome_label",
#@        "treatment",
#@        "treatment_transform",
#@        "model_treatment_scale",
#@        "reported_effect_scale",
#@        "parallel_result",
#@        "parallel_pass",
#@        "parallel_coef",
#@        "parallel_model_coef",
#@        "parallel_average_discrete_factor",
#@        "parallel_ci_low",
#@        "parallel_ci_high",
#@        "parallel_p",
#@        "did_coef",
#@        "did_model_coef",
#@        "did_average_discrete_factor",
#@        "did_ci_low",
#@        "did_ci_high",
#@        "did_p",
#@        "did_significant",
#@        "final_significant",
#@        "estimation_status",
#@        "error_message"
#@    ]
#@    return out[preferred].copy()
#@
#@
#@def build_output_readme():
#@    """Plain-language guide to the scales used in the exported tables."""
#@    return pd.DataFrame({
#@        "item": [
#@            "Main treatment form",
#@            "Robustness checks using main form",
#@            "Raw-treatment sensitivity",
#@            "coef columns",
#@            "model_coef columns",
#@            "average_discrete_factor columns",
#@            "p values",
#@        ],
#@        "description": [
#@            "log(1 + first-time asylum applicants per 1,000 residents)",
#@            "weighted, clustering, placebo, and threshold +20",
#@            "Uses the untransformed rate as an explicit functional-form sensitivity check",
#@            "Reported sample-average effect of increasing the rate by 1 applicant per 1,000 residents",
#@            "Original coefficient on the model treatment scale; for log1p checks this is the log-scale beta",
#@            "Positive factor used to convert model_coef, model SE, and model CI to the reported effect scale",
#@            "Two-sided p values from the fitted model; conversion does not change t statistics or p values",
#@        ],
#@    })
#@
#@
#@def export_significance_excel():
#@    df_all = build_significance_master_table()
#@    out_path = os.path.join(OUT_ROOT, "robustness_significance_summary.xlsx")
#@
#@    if df_all.empty:
#@        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#@            pd.DataFrame({"message": ["No summary csv files were found."]}).to_excel(
#@                writer, sheet_name="README", index=False
#@            )
#@        print("Saved:", out_path)
#@        return
#@
#@    df_overview = build_significance_overview(df_all)
#@    df_sig_only = build_significant_only_table(df_all)
#@
#@    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#@        build_output_readme().to_excel(writer, sheet_name="README", index=False)
#@        df_overview.to_excel(writer, sheet_name="overview", index=False)
#@        df_sig_only.to_excel(writer, sheet_name="significant_only", index=False)
#@        df_all.to_excel(writer, sheet_name="all_models", index=False)
#@
#@    print("Saved:", out_path)
#@
#@
#@def export_concise_significance_excel():
#@    df_all = build_significance_master_table()
#@    out_path = os.path.join(OUT_ROOT, "robustness_significance_summary_concise.xlsx")
#@
#@    if df_all.empty:
#@        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#@            pd.DataFrame({"message": ["No summary csv files were found."]}).to_excel(
#@                writer, sheet_name="README", index=False
#@            )
#@        print("Saved:", out_path)
#@        return
#@
#@    df_concise = build_concise_side_table(df_all)
#@
#@    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#@        build_output_readme().to_excel(writer, sheet_name="README", index=False)
#@        df_concise.to_excel(writer, sheet_name="concise_side_summary", index=False)
#@        df_concise[df_concise["final_significant"]].to_excel(
#@            writer, sheet_name="significant_only", index=False
#@        )
#@
#@    print("Saved:", out_path)
#@
#@
#@# ============================================================
#@# MAIN
#@# ============================================================
#@def main():
#@    safe_mkdir(OUT_ROOT)
#@
#@    print("=" * 70)
#@    print("Running robustness checks...")
#@    print("Output root:", OUT_ROOT)
#@    print("Main treatment transform:", MAIN_TREATMENT_TRANSFORM)
#@    print(
#@        "Reported effect: average change for +",
#@        DISCRETE_INCREASE_PER_1000,
#@        "applicant(s) per 1,000 residents",
#@        sep="",
#@    )
#@    print("=" * 70)
#@
#@    # 1. Weighted
#@    run_weighted_check()
#@
#@    # 2. Clustering robustness
#@    run_cluster_robustness_check()
#@
#@    # 3. Raw linear-treatment sensitivity
#@    run_raw_treatment_check()
#@
#@    # 4. Placebo
#@    run_placebo_check()
#@
#@    # 5. Threshold +20
#@    run_threshold_plus20_check()
#@
#@    # 6. Export original significance summary Excel
#@    export_significance_excel()
#@
#@    # 7. Export concise significance summary Excel
#@    export_concise_significance_excel()
#@
#@    print("Finished.")
#@    print("Robustness root:", OUT_ROOT)
#@
#@
#@if __name__ == "__main__":
#@    main()
# === END EMBEDDED sentiment_checks ===


# === BEGIN EMBEDDED sentiment_dyads ===
#@# ============================================================
#@# DID GEOPOLITICAL SENSITIVITY: S1 AND LEAVE-ONE-COUNTRY-OUT
#@# ============================================================
#@# The main entry point runs only the two requested sensitivity analyses,
#@# for both the 2015 and 2022 DID specifications, without changing source data:
#@#       S1: exclude only Russia->Ukraine and Ukraine->Russia
#@#       LOO: leave one country out at a time (all dyads involving that country)
#@# The original model-building helpers are retained so estimation logic remains
#@# identical to the supplied latest DID code.  Full-sample estimates are kept
#@# only as the fixed comparison benchmark for the sensitivity results.
#@# ============================================================
#@
#@import os
#@import re
#@import numpy as np
#@import pandas as pd
#@import matplotlib.pyplot as plt
#@from datetime import datetime
#@from linearmodels.panel import PanelOLS
#@from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator, FixedFormatter
#@
#@# ============================================================
#@# USER PATH SETTINGS
#@# ============================================================
#@INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
#@
#@combined_path = os.path.join(INPUT_DIR, "pair_results_by_period.xlsx")
#@yearly_path = os.path.join(INPUT_DIR, "pair_results_by_year.xlsx")
#@asylum_path = os.path.join(INPUT_DIR, "asylum_data.xlsx")
#@
#@DEFAULT_OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "sentiment_hate_sensitivity"))
#@# Optional environment override is useful for a non-destructive test run.
#@OUT_ROOT = os.environ.get("DID_OUT_ROOT", DEFAULT_OUT_ROOT)
#@
#@# ============================================================
#@# CHECK INPUT FILES
#@# ============================================================
#@for fp in [combined_path, yearly_path, asylum_path]:
#@    if not os.path.exists(fp):
#@        raise FileNotFoundError(f"Input file not found: {fp}")
#@
#@os.makedirs(OUT_ROOT, exist_ok=True)
#@
#@# ============================================================
#@# GENERAL SETTINGS
#@# ============================================================
#@INCLUDE_SELF = False
#@USE_WEIGHTS = False
#@
#@# Treatment functional form
#@# Raw exposure is first-time asylum applicants per 1,000 population.
#@# The regression uses log(1 + exposure) to reduce leverage from the strongly
#@# right-skewed rate distribution and allow diminishing marginal effects.
#@TREATMENT_TRANSFORM = "log1p"
#@
#@# Report every coefficient as the sample-average discrete effect of increasing
#@# the original exposure rate by 1 applicant per 1,000 residents.  The
#@# regression is still estimated with log(1 + rate); this is only a transparent
#@# post-estimation conversion back to the original, reader-friendly unit.
#@REPORT_AVERAGE_DISCRETE_EFFECT = True
#@DISCRETE_INCREASE_PER_1000 = 1.0
#@
#@# 2022 DID settings
#@DID2022_MIN_COUNT = 30
#@DID2022_OUTCOME_SPECS = [
#@    {"ycol": "mean_sentiment_score"},
#@    {"ycol": "mean_hate_speech_score"},
#@]
#@DID2022_TREAT_SPECS = [
#@    {"treat_col": "non-Ukrainian_first_time_applicant"},
#@    {"treat_col": "first_time_applicants_ukrainian"},
#@]
#@
#@# 2015 crisis DID settings
#@CRISIS2015_MIN_COUNT = 20
#@CRISIS2015_OUTCOME_SPECS = [
#@    {"ycol": "mean_sentiment_score"},
#@    {"ycol": "mean_hate_speech_score"},
#@]
#@CRISIS2015_TREAT_SPECS = [
#@    {"treat_col": "first_time_applicants"},
#@]
#@
#@# Geopolitical sensitivity settings.  These analyses supplement rather than
#@# replace the full-sample estimates.
#@RUN_GEOPOLITICAL_SENSITIVITY_2022 = True
#@RUN_GEOPOLITICAL_SENSITIVITY_2015 = True
#@
#@SAMPLE_FULL = "full_sample"
#@SAMPLE_S1 = "S1_exclude_direct_Russia_Ukraine_dyads"
#@SAMPLE_S3 = "S3_leave_one_country_out"
#@
#@# mentioned_country names are already lower-cased by clean_country_name().  Include the
#@# common UNHCR variant so the exclusion remains correct if either label occurs.
#@RUSSIA_NAMES = {"russia", "russian federation"}
#@UKRAINE_NAMES = {"ukraine"}
#@
#@# In-memory caches avoid reading the same Excel sheets hundreds of times in the
#@# leave-one-country-out loop.  Every returned DataFrame is copied before use.
#@_PERIOD_PANEL_CACHE = {}
#@_YEARLY_LONG_CACHE = {}
#@_PRE_PANEL_2022_CACHE = {}
#@_PRE_PANEL_2015_CACHE = {}
#@_TREATMENT_CACHE = {}
#@
#@# ============================================================
#@# GLOBAL FIGURE STYLE
#@# ============================================================
#@plt.rcParams["font.family"] = "Arial"
#@plt.rcParams["font.size"] = 16
#@plt.rcParams["axes.unicode_minus"] = False
#@plt.rcParams["pdf.fonttype"] = 42
#@plt.rcParams["ps.fonttype"] = 42
#@
#@# ============================================================
#@# COLORS
#@# ============================================================
#@SENTIMENT_COLOR = "#1A5A1E"
#@HATE_COLOR = "#F36800"
#@ZERO_LINE_COLOR = "#A7A7A7"
#@
#@# ============================================================
#@# FIGURE STYLE
#@# ============================================================
#@FIG_W = 5.5
#@FIG_H = 5.15
#@
#@MARKER_SIZE = 70
#@MARKER_EDGE_WIDTH = 1.9
#@
#@CAP_SIZE = 4.8
#@ERROR_LINE_WIDTH = 2.2
#@SPINE_WIDTH = 1.5
#@
#@TICK_SIZE = 15
#@LABEL_SIZE = 17
#@ANNOT_SIZE = 16
#@
#@# ============================================================
#@# OPTIONS
#@# ============================================================
#@SHOW_YLABEL = True
#@SHOW_POST_LABEL = True
#@SAVE_PDF = False
#@ADD_STAR_BY_CI = True
#@
#@# ============================================================
#@# SPECIAL FIGURES
#@# ============================================================
#@SPECIAL_OUTCOME = "mean_sentiment_score"
#@SPECIAL_TREATMENTS = {"first_time_applicants", "non-Ukrainian_first_time_applicant"}
#@SPECIAL_SIDE = "mentioned"
#@
#@SPECIAL_BOTTOM_LABEL = "-0.08"
#@SPECIAL_TOP_LABEL = "0.08"
#@
#@# Do not overwrite numeric tick labels with fixed text.  This must remain
#@# False when coefficient scales can change (for example after log1p treatment
#@# transformation), otherwise the displayed axis values would be misleading.
#@RELABEL_SPECIAL_YTICKS = False
#@
#@
#@# ============================================================
#@# BASIC HELPERS
#@# ============================================================
#@def safe_mkdir(path: str):
#@    os.makedirs(path, exist_ok=True)
#@
#@
#@def now_tag():
#@    return datetime.now().strftime("%Y%m%d_%H%M%S")
#@
#@
#@def clean_country_name(x):
#@    if pd.isna(x):
#@        return x
#@    x = str(x).strip()
#@    x = re.sub(r"\s+", " ", x)
#@    x = x.lower()
#@    return x
#@
#@
#@def sanitize_filename(x):
#@    """
#@    Keep spaces and parentheses.
#@    Only replace characters that are invalid in Windows filenames.
#@    """
#@    return (
#@        str(x)
#@        .replace("/", "_")
#@        .replace("\\", "_")
#@        .replace(":", "_")
#@        .replace("*", "_")
#@        .replace("?", "_")
#@        .replace('"', "_")
#@        .replace("<", "_")
#@        .replace(">", "_")
#@        .replace("|", "_")
#@    )
#@
#@
#@def extract_term_or_nan(coef_df: pd.DataFrame, term: str):
#@    sub = coef_df[coef_df["term"] == term].copy()
#@    if sub.empty:
#@        return {
#@            "term": term,
#@            "coef": np.nan,
#@            "se": np.nan,
#@            "t": np.nan,
#@            "p": np.nan,
#@            "ci_low": np.nan,
#@            "ci_high": np.nan,
#@            "model_coef": np.nan,
#@            "model_se": np.nan,
#@            "model_ci_low": np.nan,
#@            "model_ci_high": np.nan,
#@            "average_discrete_factor": np.nan,
#@        }
#@
#@    r = sub.iloc[0]
#@    return {
#@        "term": term,
#@        "coef": r["coef"],
#@        "se": r["se"],
#@        "t": r["t"],
#@        "p": r["p"],
#@        "ci_low": r["ci_low"],
#@        "ci_high": r["ci_high"],
#@        "model_coef": r.get("model_coef", r["coef"]),
#@        "model_se": r.get("model_se", r["se"]),
#@        "model_ci_low": r.get("model_ci_low", r["ci_low"]),
#@        "model_ci_high": r.get("model_ci_high", r["ci_high"]),
#@        "average_discrete_factor": r.get("average_discrete_factor", 1.0),
#@    }
#@
#@
#@def coef_table_from_res(res):
#@    return pd.DataFrame({
#@        "term": res.params.index,
#@        "coef": res.params.values,
#@        "se": res.std_errors.values,
#@        "t": res.tstats.values,
#@        "p": res.pvalues.values,
#@        "ci_low": res.conf_int().iloc[:, 0].values,
#@        "ci_high": res.conf_int().iloc[:, 1].values
#@    })
#@
#@
#@def _pair_level_treatment_values(panel: pd.DataFrame, raw_col: str):
#@    """Return one raw treatment rate per dyad after checking it is constant."""
#@    if raw_col not in panel.columns:
#@        raise ValueError(f"Missing raw treatment column for reporting: {raw_col}")
#@
#@    variation = panel.groupby("pair")[raw_col].nunique(dropna=False)
#@    if (variation > 1).any():
#@        bad = variation[variation > 1].index.tolist()[:5]
#@        raise ValueError(
#@            f"Treatment is not constant within dyad for {raw_col}; examples: {bad}"
#@        )
#@
#@    values = (
#@        panel[["pair", raw_col]]
#@        .drop_duplicates("pair")[raw_col]
#@        .pipe(pd.to_numeric, errors="coerce")
#@        .dropna()
#@        .astype(float)
#@    )
#@    if values.empty:
#@        raise ValueError(f"No usable raw treatment values in {raw_col}")
#@    if (values < 0).any():
#@        raise ValueError(f"Negative raw treatment values in {raw_col}")
#@    return values
#@
#@
#@def average_discrete_factor(panel: pd.DataFrame, raw_col: str):
#@    """
#@    Average change in the model's treatment index when the original rate rises
#@    by DISCRETE_INCREASE_PER_1000 for every dyad in the estimation sample.
#@    """
#@    values = _pair_level_treatment_values(panel, raw_col)
#@    increase = float(DISCRETE_INCREASE_PER_1000)
#@    if increase <= 0:
#@        raise ValueError("DISCRETE_INCREASE_PER_1000 must be positive")
#@
#@    if TREATMENT_TRANSFORM == "log1p":
#@        contrasts = np.log1p(values + increase) - np.log1p(values)
#@    elif TREATMENT_TRANSFORM == "raw":
#@        contrasts = np.full(len(values), increase, dtype=float)
#@    else:
#@        raise ValueError(f"Unknown TREATMENT_TRANSFORM: {TREATMENT_TRANSFORM}")
#@
#@    return float(np.mean(contrasts))
#@
#@
#@def convert_coef_table_to_average_discrete_effect(
#@    coef_df: pd.DataFrame,
#@    panel: pd.DataFrame,
#@    term_to_raw_col: dict,
#@):
#@    """
#@    Convert model-scale estimates into sample-average effects of +1 applicant
#@    per 1,000 residents. Multiplying estimate, SE, and CI by the same positive
#@    factor leaves t statistics and p values unchanged.
#@    """
#@    out = coef_df.copy()
#@    out["model_coef"] = out["coef"]
#@    out["model_se"] = out["se"]
#@    out["model_ci_low"] = out["ci_low"]
#@    out["model_ci_high"] = out["ci_high"]
#@    out["average_discrete_factor"] = 1.0
#@
#@    if not REPORT_AVERAGE_DISCRETE_EFFECT:
#@        return out
#@
#@    for term, raw_col in term_to_raw_col.items():
#@        factor = average_discrete_factor(panel, raw_col)
#@        mask = out["term"] == term
#@        if not mask.any():
#@            continue
#@        out.loc[mask, "average_discrete_factor"] = factor
#@        for col in ["coef", "se", "ci_low", "ci_high"]:
#@            out.loc[mask, col] = out.loc[mask, col] * factor
#@
#@    return out
#@
#@
#@def keep_common_pairs(panel: pd.DataFrame, periods):
#@    tmp = panel.groupby("pair")["period"].nunique()
#@    valid_pairs = tmp[tmp == len(periods)].index.tolist()
#@    return panel[panel["pair"].isin(valid_pairs)].copy(), valid_pairs
#@
#@
#@def _is_russia(series: pd.Series):
#@    return series.astype(str).str.lower().isin(RUSSIA_NAMES)
#@
#@
#@def _is_ukraine(series: pd.Series):
#@    return series.astype(str).str.lower().isin(UKRAINE_NAMES)
#@
#@
#@def apply_sensitivity_filter(
#@    panel: pd.DataFrame,
#@    sample_spec: str = SAMPLE_FULL,
#@    omitted_country=None,
#@):
#@    """
#@    Return a filtered copy of an already constructed analysis panel.
#@
#@    The input panel and all source files remain unchanged.  S3 removes every
#@    dyad in which the omitted country appears on either side.  This conservative
#@    definition tests the country's total influence on the dyadic estimate.
#@    """
#@    out = panel.copy(deep=True)
#@
#@    if sample_spec == SAMPLE_FULL:
#@        mask = pd.Series(False, index=out.index)
#@
#@    elif sample_spec == SAMPLE_S1:
#@        origin_russia = _is_russia(out["origin"])
#@        origin_ukraine = _is_ukraine(out["origin"])
#@        mentioned_russia = _is_russia(out["mentioned"])
#@        mentioned_ukraine = _is_ukraine(out["mentioned"])
#@        mask = (
#@            (origin_russia & mentioned_ukraine)
#@            | (origin_ukraine & mentioned_russia)
#@        )
#@
#@    elif sample_spec == SAMPLE_S3:
#@        if omitted_country is None or pd.isna(omitted_country):
#@            raise ValueError("S3 requires omitted_country")
#@        omitted = clean_country_name(omitted_country)
#@        mask = (out["origin"] == omitted) | (out["mentioned"] == omitted)
#@
#@    else:
#@        raise ValueError(f"Unknown sample_spec: {sample_spec}")
#@
#@    return out.loc[~mask].copy().reset_index(drop=True)
#@
#@
#@def analysis_country_universe(*panels):
#@    countries = set()
#@    for panel in panels:
#@        if panel is None or panel.empty:
#@            continue
#@        countries.update(panel["origin"].dropna().astype(str).tolist())
#@        countries.update(panel["mentioned"].dropna().astype(str).tolist())
#@    return sorted(countries)
#@
#@
#@def add_sensitivity_metadata(
#@    summary: dict,
#@    crisis: str,
#@    sample_spec: str,
#@    omitted_country,
#@    pre_before: pd.DataFrame,
#@    post_before: pd.DataFrame,
#@    pre_after: pd.DataFrame,
#@    post_after: pd.DataFrame,
#@):
#@    """Attach transparent sample-audit fields to every sensitivity estimate."""
#@    row = dict(summary)
#@    row.update({
#@        "crisis": str(crisis),
#@        "sample_spec": str(sample_spec),
#@        "omitted_country": (
#@            "" if omitted_country is None else clean_country_name(omitted_country)
#@        ),
#@        "status": "ok",
#@        "error_message": "",
#@        "n_rows_pre_before_filter": int(len(pre_before)),
#@        "n_rows_pre_after_filter": int(len(pre_after)),
#@        "n_rows_post_before_filter": int(len(post_before)),
#@        "n_rows_post_after_filter": int(len(post_after)),
#@        "n_pairs_pre_before_filter": int(pre_before["pair"].nunique()),
#@        "n_pairs_pre_after_filter": int(pre_after["pair"].nunique()),
#@        "n_pairs_post_before_filter": int(post_before["pair"].nunique()),
#@        "n_pairs_post_after_filter": int(post_after["pair"].nunique()),
#@        "n_unique_countries_pre_after_filter": int(
#@            len(analysis_country_universe(pre_after))
#@        ),
#@        "n_unique_countries_post_after_filter": int(
#@            len(analysis_country_universe(post_after))
#@        ),
#@        "n_rows_pre_removed": int(len(pre_before) - len(pre_after)),
#@        "n_rows_post_removed": int(len(post_before) - len(post_after)),
#@        "n_pairs_pre_removed": int(
#@            pre_before["pair"].nunique() - pre_after["pair"].nunique()
#@        ),
#@        "n_pairs_post_removed": int(
#@            post_before["pair"].nunique() - post_after["pair"].nunique()
#@        ),
#@    })
#@    return row
#@
#@
#@# ============================================================
#@# FIGURE HELPERS
#@# ============================================================
#@def get_line_color(outcome_value):
#@    if str(outcome_value) == "mean_sentiment_score":
#@        return SENTIMENT_COLOR
#@    return HATE_COLOR
#@
#@
#@def unified_x():
#@    return [0, 1, 2]
#@
#@
#@def ci_excludes_zero(ci_low, ci_high):
#@    if pd.isna(ci_low) or pd.isna(ci_high):
#@        return False
#@    return (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
#@
#@
#@def format_coef_label(value, ci_low=None, ci_high=None):
#@    """
#@    Display the reported average discrete effect.
#@    Example: ΔY = -0.022*
#@    """
#@    if pd.isna(value):
#@        return ""
#@
#@    star = ""
#@    if ADD_STAR_BY_CI and ci_excludes_zero(ci_low, ci_high):
#@        star = "*"
#@
#@    return f"ΔY = {value:.3f}{star}"
#@
#@
#@def dynamic_formatter(x, pos):
#@    """
#@    Adapt tick precision to the magnitude of the axis.
#@
#@    Important:
#@    The y-axis zero label is always displayed as 0.00.
#@    """
#@    if np.isclose(x, 0, atol=1e-12):
#@        return "0.00"
#@
#@    ax_abs = abs(x)
#@
#@    if ax_abs >= 1:
#@        return f"{x:.1f}"
#@    elif ax_abs >= 0.1:
#@        return f"{x:.2f}"
#@    else:
#@        return f"{x:.3f}"
#@
#@
#@def nice_ylim_raw(ci_low, ci_high):
#@    """
#@    Force y-axis to be symmetric around zero.
#@    """
#@    vals = [v for v in ci_low + ci_high if pd.notna(v)]
#@
#@    if not vals:
#@        return (-0.05, 0.05)
#@
#@    max_abs = max(abs(min(vals)), abs(max(vals)), 0)
#@
#@    limit = max_abs * 1.22
#@
#@    min_half_span = 0.03
#@    if limit < min_half_span:
#@        limit = min_half_span
#@
#@    return -limit, limit
#@
#@
#@def style_axis(ax):
#@    ax.axhline(
#@        0,
#@        linestyle="--",
#@        linewidth=1.6,
#@        color=ZERO_LINE_COLOR,
#@        zorder=1
#@    )
#@
#@    ax.set_xlim(-0.30, 2.55)
#@
#@    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
#@    ax.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
#@
#@    ax.tick_params(
#@        axis="both",
#@        labelsize=TICK_SIZE,
#@        width=1.25,
#@        length=4.2,
#@        top=False,
#@        right=False
#@    )
#@
#@    ax.spines["top"].set_visible(False)
#@    ax.spines["right"].set_visible(False)
#@
#@    ax.spines["left"].set_linewidth(SPINE_WIDTH)
#@    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
#@
#@    ax.grid(False)
#@
#@
#@def use_special_tick_labels(row, side, prefix):
#@    """
#@    Special figures:
#@    1. crisis2015_sentiment_first-time asylum applicant_mentioned.png
#@    2. 2022_sentiment_first-time asylum applicant_mentioned.png
#@    """
#@    outcome = str(row["outcome"])
#@    treatment = str(row["treatment"])
#@
#@    cond1 = (
#@        prefix == "crisis2015"
#@        and outcome == SPECIAL_OUTCOME
#@        and treatment in SPECIAL_TREATMENTS
#@        and side == SPECIAL_SIDE
#@    )
#@
#@    cond2 = (
#@        prefix == "2022"
#@        and outcome == SPECIAL_OUTCOME
#@        and treatment in SPECIAL_TREATMENTS
#@        and side == SPECIAL_SIDE
#@    )
#@
#@    return cond1 or cond2
#@
#@
#@def relabel_top_bottom_yticks(ax, bottom_label="-0.08", top_label="0.08"):
#@    ymin, ymax = ax.get_ylim()
#@    ticks = ax.get_yticks()
#@    visible_ticks = [t for t in ticks if ymin <= t <= ymax]
#@
#@    if len(visible_ticks) < 2:
#@        return
#@
#@    bottom_tick = visible_ticks[0]
#@    top_tick = visible_ticks[-1]
#@
#@    labels = []
#@    for t in visible_ticks:
#@        if np.isclose(t, bottom_tick):
#@            labels.append(bottom_label)
#@        elif np.isclose(t, top_tick):
#@            labels.append(top_label)
#@        else:
#@            labels.append(dynamic_formatter(t, None))
#@
#@    ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
#@    ax.yaxis.set_major_formatter(FixedFormatter(labels))
#@
#@
#@def draw_coef_point(
#@    ax,
#@    x_value,
#@    y_value,
#@    yerr_lower,
#@    yerr_upper,
#@    ci_low,
#@    ci_high,
#@    color,
#@    zorder=4
#@):
#@    """
#@    Draw CI and coefficient point separately.
#@
#@    Significant = filled circle.
#@    Non-significant = hollow circle.
#@    """
#@    if pd.isna(y_value):
#@        return
#@
#@    if pd.notna(yerr_lower) and pd.notna(yerr_upper):
#@        ax.errorbar(
#@            [x_value],
#@            [y_value],
#@            yerr=[[yerr_lower], [yerr_upper]],
#@            fmt="none",
#@            ecolor=color,
#@            elinewidth=ERROR_LINE_WIDTH,
#@            capsize=0,
#@            alpha=1.0,
#@            zorder=zorder
#@        )
#@
#@    is_sig = ci_excludes_zero(ci_low, ci_high)
#@
#@    ax.scatter(
#@        [x_value],
#@        [y_value],
#@        s=MARKER_SIZE,
#@        facecolors=color if is_sig else "white",
#@        edgecolors=color,
#@        linewidths=MARKER_EDGE_WIDTH,
#@        alpha=1.0,
#@        zorder=zorder + 1
#@    )
#@
#@
#@def outcome_to_filename_label(outcome):
#@    outcome = str(outcome)
#@
#@    if outcome == "mean_sentiment_score":
#@        return "sentiment"
#@    elif outcome == "mean_hate_speech_score":
#@        return "hate speech"
#@
#@    return sanitize_filename(outcome)
#@
#@
#@def treatment_to_filename_label(treatment):
#@    treatment = str(treatment)
#@
#@    if treatment == "non-Ukrainian_first_time_applicant":
#@        return "non-Ukrainian first-time asylum applicant"
#@    if treatment == "first_time_applicants":
#@        return "first-time asylum applicant"
#@    elif treatment == "first_time_applicants_total":
#@        return "first-time asylum applicant (alternative total field)"
#@    elif treatment == "first_time_applicants_ukrainian":
#@        return "first-time asylum applicant (Ukrainians)"
#@
#@    return sanitize_filename(treatment)
#@
#@
#@def make_figure_filename(prefix, outcome, treatment, side):
#@    """
#@    Required examples:
#@    2022_hate speech_first-time asylum applicant (Ukrainians)_origin.png
#@    crisis2015_sentiment_first-time asylum applicant_mentioned.png
#@    """
#@    crisis_label = prefix
#@    outcome_label = outcome_to_filename_label(outcome)
#@    treatment_label = treatment_to_filename_label(treatment)
#@    side_label = str(side)
#@
#@    fname = f"{crisis_label}_{outcome_label}_{treatment_label}_{side_label}.png"
#@    return sanitize_filename(fname)
#@
#@
#@# ============================================================
#@# LOAD PERIOD PANEL
#@# ============================================================
#@def read_panel_from_period_file(combined_path: str, sheets, ycol: str, min_count: int, include_self: bool):
#@    cache_key = (
#@        os.path.abspath(combined_path),
#@        tuple(str(s) for s in sheets),
#@        str(ycol),
#@        int(min_count),
#@        bool(include_self),
#@    )
#@    if cache_key in _PERIOD_PANEL_CACHE:
#@        return _PERIOD_PANEL_CACHE[cache_key].copy(deep=True)
#@
#@    data_dict = pd.read_excel(combined_path, sheet_name=sheets)
#@    all_rows = []
#@
#@    for sh in sheets:
#@        df = data_dict[sh].copy()
#@
#@        need_cols = ["origin_country", "mentioned_country", "post_count", ycol]
#@        missing = [c for c in need_cols if c not in df.columns]
#@        if missing:
#@            raise ValueError(f"[{sh}] missing columns: {missing}")
#@
#@        df["origin_country"] = df["origin_country"].map(clean_country_name)
#@        df["mentioned_country"] = df["mentioned_country"].map(clean_country_name)
#@        df["post_count"] = pd.to_numeric(df["post_count"], errors="coerce")
#@        df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
#@
#@        df = df[df["post_count"] >= min_count].copy()
#@        df = df[~df[ycol].isna()].copy()
#@
#@        df = df.rename(columns={
#@            "origin_country": "origin",
#@            "mentioned_country": "mentioned",
#@            ycol: "Y"
#@        })
#@
#@        if not include_self:
#@            df = df[df["origin"] != df["mentioned"]].copy()
#@
#@        df["period"] = sh
#@        df["pair"] = df["origin"] + "->" + df["mentioned"]
#@        df = df.drop_duplicates(subset=["pair", "period"])
#@
#@        all_rows.append(df[["pair", "origin", "mentioned", "period", "Y", "post_count"]])
#@
#@    panel = pd.concat(all_rows, ignore_index=True)
#@    _PERIOD_PANEL_CACHE[cache_key] = panel.copy(deep=True)
#@    return panel.copy(deep=True)
#@
#@
#@# ============================================================
#@# YEARLY FILE HELPERS
#@# ============================================================
#@def _find_year_col(df):
#@    for c in ["Year", "year", "YEAR"]:
#@        if c in df.columns:
#@            return c
#@    return None
#@
#@
#@def _standardize_yearly_df(df: pd.DataFrame, year_value=None):
#@    df = df.copy()
#@
#@    need_some_country_cols = ["origin_country", "mentioned_country"]
#@    missing_country = [c for c in need_some_country_cols if c not in df.columns]
#@    if missing_country:
#@        raise ValueError(f"Yearly data missing columns: {missing_country}")
#@
#@    df["origin_country"] = df["origin_country"].map(clean_country_name)
#@    df["mentioned_country"] = df["mentioned_country"].map(clean_country_name)
#@
#@    needed_numeric = [
#@        "positive_post_count", "neutral_post_count", "negative_post_count",
#@        "hate_speech_post_count", "normal_post_count", "offensive_post_count"
#@    ]
#@    missing_numeric = [c for c in needed_numeric if c not in df.columns]
#@    if missing_numeric:
#@        raise ValueError(f"Yearly data missing columns: {missing_numeric}")
#@
#@    for c in needed_numeric:
#@        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
#@
#@    if year_value is not None:
#@        df["year"] = int(year_value)
#@    else:
#@        yc = _find_year_col(df)
#@        if yc is None:
#@            raise ValueError("Yearly file must have per-year sheets or a Year/year column.")
#@        df["year"] = pd.to_numeric(df[yc], errors="coerce")
#@
#@    df = df.dropna(subset=["year"]).copy()
#@    df["year"] = df["year"].astype(int)
#@
#@    return df
#@
#@
#@def load_yearly_long(yearly_path: str, target_years):
#@    cache_key = (os.path.abspath(yearly_path), tuple(int(y) for y in target_years))
#@    if cache_key in _YEARLY_LONG_CACHE:
#@        return _YEARLY_LONG_CACHE[cache_key].copy(deep=True)
#@
#@    xls = pd.ExcelFile(yearly_path)
#@    sheet_names = xls.sheet_names
#@
#@    year_sheets_found = []
#@    for sh in sheet_names:
#@        s = str(sh).strip()
#@        if s.isdigit() and int(s) in target_years:
#@            year_sheets_found.append(int(s))
#@
#@    if set(year_sheets_found) == set(target_years):
#@        parts = []
#@        for yy in target_years:
#@            tmp = pd.read_excel(yearly_path, sheet_name=str(yy))
#@            tmp = _standardize_yearly_df(tmp, year_value=yy)
#@            parts.append(tmp)
#@        out = pd.concat(parts, ignore_index=True)
#@        _YEARLY_LONG_CACHE[cache_key] = out.copy(deep=True)
#@        return out.copy(deep=True)
#@
#@    for sh in sheet_names:
#@        tmp = pd.read_excel(yearly_path, sheet_name=sh)
#@        yc = _find_year_col(tmp)
#@        needed = [
#@            "origin_country", "mentioned_country",
#@            "positive_post_count", "neutral_post_count", "negative_post_count",
#@            "hate_speech_post_count", "normal_post_count", "offensive_post_count"
#@        ]
#@        if yc is not None and all(c in tmp.columns for c in needed):
#@            tmp = _standardize_yearly_df(tmp, year_value=None)
#@            tmp = tmp[tmp["year"].isin(target_years)].copy()
#@            _YEARLY_LONG_CACHE[cache_key] = tmp.copy(deep=True)
#@            return tmp.copy(deep=True)
#@
#@    raise ValueError("Cannot parse yearly file correctly.")
#@
#@
#@# ============================================================
#@# BUILD PRE PANEL - 2022 DID VERSION
#@# ============================================================
#@def _map_2022_pre_period(y):
#@    y = int(y)
#@    if 2010 <= y <= 2014:
#@        return "2010-2014"
#@    elif 2015 <= y <= 2021:
#@        return "2015-2021"
#@    return np.nan
#@
#@
#@def build_pre_panel_2022_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
#@    cache_key = (
#@        os.path.abspath(yearly_path), str(ycol), int(min_count), bool(include_self)
#@    )
#@    if cache_key in _PRE_PANEL_2022_CACHE:
#@        return _PRE_PANEL_2022_CACHE[cache_key].copy(deep=True)
#@
#@    target_years = list(range(2010, 2022))
#@    df = load_yearly_long(yearly_path, target_years).copy()
#@
#@    df = df.rename(columns={"origin_country": "origin", "mentioned_country": "mentioned"})
#@
#@    if not include_self:
#@        df = df[df["origin"] != df["mentioned"]].copy()
#@
#@    df["pair"] = df["origin"] + "->" + df["mentioned"]
#@    df["period"] = df["year"].map(_map_2022_pre_period)
#@    df = df.dropna(subset=["period"]).copy()
#@
#@    agg = (
#@        df.groupby(["pair", "origin", "mentioned", "period"], as_index=False)
#@        .agg({
#@            "positive_post_count": "sum",
#@            "neutral_post_count": "sum",
#@            "negative_post_count": "sum",
#@            "hate_speech_post_count": "sum",
#@            "normal_post_count": "sum",
#@            "offensive_post_count": "sum",
#@            "year": "nunique"
#@        })
#@        .rename(columns={"year": "n_years"})
#@    )
#@
#@    sent_den = agg["positive_post_count"] + agg["neutral_post_count"] + agg["negative_post_count"]
#@    hate_den = agg["hate_speech_post_count"] + agg["normal_post_count"] + agg["offensive_post_count"]
#@
#@    if ycol == "mean_sentiment_score":
#@        agg["raw_num"] = agg["positive_post_count"] - agg["negative_post_count"]
#@        agg["raw_den"] = sent_den
#@        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
#@        agg["post_count"] = agg["raw_den"]
#@    elif ycol == "mean_hate_speech_score":
#@        agg["raw_num"] = agg["hate_speech_post_count"] * 2 + agg["offensive_post_count"]
#@        agg["raw_den"] = hate_den
#@        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
#@        agg["post_count"] = agg["raw_den"]
#@    else:
#@        raise ValueError(f"Unsupported ycol: {ycol}")
#@
#@    agg = agg[agg["post_count"] >= min_count].copy()
#@    agg = agg[~agg["Y"].isna()].copy()
#@
#@    result = agg[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den", "n_years"]]
#@    _PRE_PANEL_2022_CACHE[cache_key] = result.copy(deep=True)
#@    return result.copy(deep=True)
#@
#@
#@# ============================================================
#@# BUILD PRE PANEL - 2015 CRISIS VERSION
#@# ============================================================
#@def _map_2015_pre_period(y):
#@    y = int(y)
#@    if y in [2010, 2011, 2012]:
#@        return "2010-2012"
#@    elif y in [2013, 2014]:
#@        return "2013-2014"
#@    return np.nan
#@
#@
#@def build_pre_panel_2015_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
#@    cache_key = (
#@        os.path.abspath(yearly_path), str(ycol), int(min_count), bool(include_self)
#@    )
#@    if cache_key in _PRE_PANEL_2015_CACHE:
#@        return _PRE_PANEL_2015_CACHE[cache_key].copy(deep=True)
#@
#@    target_years = [2010, 2011, 2012, 2013, 2014]
#@    df = load_yearly_long(yearly_path, target_years).copy()
#@
#@    df = df.rename(columns={"origin_country": "origin", "mentioned_country": "mentioned"})
#@
#@    if not include_self:
#@        df = df[df["origin"] != df["mentioned"]].copy()
#@
#@    df["pair"] = df["origin"] + "->" + df["mentioned"]
#@    df["period"] = df["year"].map(_map_2015_pre_period)
#@    df = df.dropna(subset=["period"]).copy()
#@
#@    agg = (
#@        df.groupby(["pair", "origin", "mentioned", "period"], as_index=False)
#@        .agg({
#@            "positive_post_count": "sum",
#@            "neutral_post_count": "sum",
#@            "negative_post_count": "sum",
#@            "hate_speech_post_count": "sum",
#@            "normal_post_count": "sum",
#@            "offensive_post_count": "sum",
#@            "year": "nunique"
#@        })
#@        .rename(columns={"year": "n_years"})
#@    )
#@
#@    sent_den = agg["positive_post_count"] + agg["neutral_post_count"] + agg["negative_post_count"]
#@    hate_den = agg["hate_speech_post_count"] + agg["normal_post_count"] + agg["offensive_post_count"]
#@
#@    if ycol == "mean_sentiment_score":
#@        agg["raw_num"] = agg["positive_post_count"] - agg["negative_post_count"]
#@        agg["raw_den"] = sent_den
#@        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
#@        agg["post_count"] = agg["raw_den"]
#@    elif ycol == "mean_hate_speech_score":
#@        agg["raw_num"] = agg["hate_speech_post_count"] * 2 + agg["offensive_post_count"]
#@        agg["raw_den"] = hate_den
#@        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
#@        agg["post_count"] = agg["raw_den"]
#@    else:
#@        raise ValueError(f"Unsupported ycol: {ycol}")
#@
#@    agg = agg[agg["post_count"] >= min_count].copy()
#@    agg = agg[~agg["Y"].isna()].copy()
#@
#@    result = agg[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den", "n_years"]]
#@    _PRE_PANEL_2015_CACHE[cache_key] = result.copy(deep=True)
#@    return result.copy(deep=True)
#@
#@
#@# ============================================================
#@# TREATMENT HELPERS
#@# ============================================================
#@def make_treat_k(asylum_path: str, year: int, treat_col: str):
#@    cache_key = (
#@        os.path.abspath(asylum_path), int(year), str(treat_col),
#@        str(TREATMENT_TRANSFORM)
#@    )
#@    if cache_key in _TREATMENT_CACHE:
#@        return _TREATMENT_CACHE[cache_key].copy(deep=True)
#@
#@    a = pd.read_excel(asylum_path, sheet_name=str(year)).copy()
#@
#@    need_cols = ["host_country", treat_col, "population"]
#@    missing = [c for c in need_cols if c not in a.columns]
#@    if missing:
#@        raise ValueError(f"[asylum {year}] missing columns: {missing}")
#@
#@    a = a.rename(columns={"host_country": "country"})
#@    a["country"] = a["country"].map(clean_country_name)
#@
#@    a[treat_col] = pd.to_numeric(a[treat_col], errors="coerce")
#@    a["population"] = pd.to_numeric(a["population"], errors="coerce")
#@
#@    treat_k_col = f"treat_k_{year}"
#@    raw_treat_k_col = f"raw_treat_k_{year}"
#@    raw_rate = 1000.0 * a[treat_col] / a["population"]
#@    a[raw_treat_k_col] = raw_rate
#@
#@    if (raw_rate.dropna() < 0).any():
#@        raise ValueError(
#@            f"[{treat_col}, {year}] negative treatment rates are incompatible "
#@            "with the log1p transformation."
#@        )
#@
#@    if TREATMENT_TRANSFORM == "log1p":
#@        a[treat_k_col] = np.log1p(raw_rate)
#@    elif TREATMENT_TRANSFORM == "raw":
#@        a[treat_k_col] = raw_rate
#@    else:
#@        raise ValueError(
#@            f"Unknown TREATMENT_TRANSFORM: {TREATMENT_TRANSFORM}"
#@        )
#@
#@    result = a[["country", treat_k_col, raw_treat_k_col]]
#@    _TREATMENT_CACHE[cache_key] = result.copy(deep=True)
#@    return result.copy(deep=True)
#@
#@
#@def attach_treatment_two_sides(panel: pd.DataFrame, asylum_path: str, treat_col: str, treat_year: int):
#@    t = make_treat_k(asylum_path, treat_year, treat_col=treat_col)
#@    tk = f"treat_k_{treat_year}"
#@    raw_tk = f"raw_treat_k_{treat_year}"
#@
#@    panel = panel.merge(
#@        t.rename(columns={
#@            "country": "origin",
#@            tk: "treat_origin",
#@            raw_tk: "raw_treat_origin",
#@        }),
#@        on="origin",
#@        how="left"
#@    )
#@    panel = panel.merge(
#@        t.rename(columns={
#@            "country": "mentioned",
#@            tk: "treat_mentioned",
#@            raw_tk: "raw_treat_mentioned",
#@        }),
#@        on="mentioned",
#@        how="left"
#@    )
#@
#@    panel["treat_origin"] = pd.to_numeric(panel["treat_origin"], errors="coerce")
#@    panel["treat_mentioned"] = pd.to_numeric(panel["treat_mentioned"], errors="coerce")
#@    panel["raw_treat_origin"] = pd.to_numeric(panel["raw_treat_origin"], errors="coerce")
#@    panel["raw_treat_mentioned"] = pd.to_numeric(panel["raw_treat_mentioned"], errors="coerce")
#@    return panel
#@
#@
#@def attach_dyadic_treat_k_2015(panel: pd.DataFrame, asylum_path: str, treat_col: str):
#@    t15 = make_treat_k(asylum_path, 2015, treat_col=treat_col)
#@
#@    panel = panel.merge(
#@        t15.rename(columns={
#@            "country": "origin",
#@            "treat_k_2015": "treat_origin_2015",
#@            "raw_treat_k_2015": "raw_treat_origin_2015",
#@        }),
#@        on="origin", how="left"
#@    )
#@    panel = panel.merge(
#@        t15.rename(columns={
#@            "country": "mentioned",
#@            "treat_k_2015": "treat_mentioned_2015",
#@            "raw_treat_k_2015": "raw_treat_mentioned_2015",
#@        }),
#@        on="mentioned", how="left"
#@    )
#@
#@    panel["treat_origin_2015"] = pd.to_numeric(panel["treat_origin_2015"], errors="coerce")
#@    panel["treat_mentioned_2015"] = pd.to_numeric(panel["treat_mentioned_2015"], errors="coerce")
#@    panel["raw_treat_origin_2015"] = pd.to_numeric(panel["raw_treat_origin_2015"], errors="coerce")
#@    panel["raw_treat_mentioned_2015"] = pd.to_numeric(panel["raw_treat_mentioned_2015"], errors="coerce")
#@    return panel
#@
#@
#@# ============================================================
#@# VARIABLE BUILDERS
#@# ============================================================
#@def build_pretrend_vars_2022(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2010-2014": 0, "2015-2021": 1})
#@    panel["PrePeriod"] = (panel["period"] == "2010-2014").astype(int)
#@    panel["PRE_origin"] = panel["treat_origin"] * panel["PrePeriod"]
#@    panel["PRE_mentioned"] = panel["treat_mentioned"] * panel["PrePeriod"]
#@    return panel
#@
#@
#@def build_post_did_vars_2022(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2015-2021": 0, "2022-2024": 1})
#@    panel["PostPeriod"] = (panel["period"] == "2022-2024").astype(int)
#@    panel["POST_origin"] = panel["treat_origin"] * panel["PostPeriod"]
#@    panel["POST_mentioned"] = panel["treat_mentioned"] * panel["PostPeriod"]
#@    return panel
#@
#@
#@def build_pretrend_vars_2015(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2010-2012": 0, "2013-2014": 1})
#@    panel["PrePeriod"] = (panel["period"] == "2010-2012").astype(int)
#@    panel["PRE_origin"] = panel["treat_origin_2015"] * panel["PrePeriod"]
#@    panel["PRE_mentioned"] = panel["treat_mentioned_2015"] * panel["PrePeriod"]
#@    return panel
#@
#@
#@def build_post_did_vars_2015(panel: pd.DataFrame):
#@    panel = panel.copy()
#@    panel["period_id"] = panel["period"].map({"2013-2014": 0, "2015-2021": 1})
#@    panel["Post2015"] = (panel["period"] == "2015-2021").astype(int)
#@    panel["POST_origin"] = panel["treat_origin_2015"] * panel["Post2015"]
#@    panel["POST_mentioned"] = panel["treat_mentioned_2015"] * panel["Post2015"]
#@    return panel
#@
#@
#@# ============================================================
#@# REGRESSION
#@# ============================================================
#@def run_panel_ols(panel: pd.DataFrame, xcols, cluster_mode: str = "pair", use_weights: bool = False):
#@    df = panel.copy().set_index(["pair", "period_id"]).sort_index()
#@
#@    y = df["Y"]
#@    X = df[xcols]
#@
#@    model_kwargs = dict(
#@        dependent=y,
#@        exog=X,
#@        entity_effects=True,
#@        time_effects=True,
#@        drop_absorbed=True
#@    )
#@
#@    if use_weights:
#@        model_kwargs["weights"] = pd.to_numeric(df["post_count"], errors="coerce").astype(float)
#@
#@    model = PanelOLS(**model_kwargs)
#@
#@    if cluster_mode == "pair":
#@        res = model.fit(cov_type="clustered", cluster_entity=True)
#@    elif cluster_mode == "origin_mentioned":
#@        clusters = pd.DataFrame({
#@            "origin": df["origin"].astype("category"),
#@            "mentioned": df["mentioned"].astype("category")
#@        }, index=df.index)
#@        res = model.fit(cov_type="clustered", clusters=clusters)
#@    else:
#@        raise ValueError("Unknown cluster_mode")
#@
#@    return res, df
#@
#@
#@# ============================================================
#@# FINAL SUMMARY ROWS
#@# ============================================================
#@def make_final_summary_row_2022(ycol, treat_col, pre_coef_df, post_coef_df, pre_panel, post_panel):
#@    pre_origin = extract_term_or_nan(pre_coef_df, "PRE_origin")
#@    pre_mentioned = extract_term_or_nan(pre_coef_df, "PRE_mentioned")
#@    post_origin = extract_term_or_nan(post_coef_df, "POST_origin")
#@    post_mentioned = extract_term_or_nan(post_coef_df, "POST_mentioned")
#@
#@    origin_parallel = "pass" if pd.notna(pre_origin["p"]) and pre_origin["p"] >= 0.05 else "fail"
#@    mentioned_parallel = "pass" if pd.notna(pre_mentioned["p"]) and pre_mentioned["p"] >= 0.05 else "fail"
#@    overall_parallel = "pass_both_sides" if origin_parallel == "pass" and mentioned_parallel == "pass" else "fail_at_least_one_side"
#@
#@    return {
#@        "outcome": ycol,
#@        "treatment": treat_col,
#@        "model_treatment_scale": "log(1 + first-time asylum applicants per 1,000 population)",
#@        "reported_effect_scale": "sample-average discrete effect of +1 applicant per 1,000 residents",
#@        "estimator": "two-period pair and time fixed-effects DID",
#@        "covariance": "pair-clustered standard errors",
#@
#@        "n_pairs_pre_total": pre_panel["pair"].nunique(),
#@        "n_obs_pre": len(pre_panel),
#@        "n_pairs_post_common": post_panel["pair"].nunique(),
#@        "n_obs_post": len(post_panel),
#@
#@        "origin_parallel": origin_parallel,
#@        "mentioned_parallel": mentioned_parallel,
#@        "overall_parallel": overall_parallel,
#@
#@        "origin_pre_period": "2010-2014",
#@        "origin_pre_event_time": -2,
#@        "origin_pre_coef_2010_2014_vs_2015_2021": pre_origin["coef"],
#@        "origin_pre_model_log1p_coef_2010_2014_vs_2015_2021": pre_origin["model_coef"],
#@        "origin_pre_average_discrete_factor": pre_origin["average_discrete_factor"],
#@        "origin_pre_se_2010_2014_vs_2015_2021": pre_origin["se"],
#@        "origin_pre_p_2010_2014_vs_2015_2021": pre_origin["p"],
#@        "origin_pre_ci_low_2010_2014_vs_2015_2021": pre_origin["ci_low"],
#@        "origin_pre_ci_high_2010_2014_vs_2015_2021": pre_origin["ci_high"],
#@
#@        "origin_base_period": "2015-2021",
#@        "origin_base_event_time": -1,
#@        "origin_base_coef_2015_2021": 0.0,
#@        "origin_base_se_2015_2021": 0.0,
#@        "origin_base_p_2015_2021": np.nan,
#@        "origin_base_ci_low_2015_2021": 0.0,
#@        "origin_base_ci_high_2015_2021": 0.0,
#@
#@        "origin_post_period": "2022-2024",
#@        "origin_post_event_time": 1,
#@        "origin_post_coef_2022_2024_vs_2015_2021": post_origin["coef"],
#@        "origin_post_model_log1p_coef_2022_2024_vs_2015_2021": post_origin["model_coef"],
#@        "origin_post_average_discrete_factor": post_origin["average_discrete_factor"],
#@        "origin_post_se_2022_2024_vs_2015_2021": post_origin["se"],
#@        "origin_post_p_2022_2024_vs_2015_2021": post_origin["p"],
#@        "origin_post_ci_low_2022_2024_vs_2015_2021": post_origin["ci_low"],
#@        "origin_post_ci_high_2022_2024_vs_2015_2021": post_origin["ci_high"],
#@
#@        "mentioned_pre_period": "2010-2014",
#@        "mentioned_pre_event_time": -2,
#@        "mentioned_pre_coef_2010_2014_vs_2015_2021": pre_mentioned["coef"],
#@        "mentioned_pre_model_log1p_coef_2010_2014_vs_2015_2021": pre_mentioned["model_coef"],
#@        "mentioned_pre_average_discrete_factor": pre_mentioned["average_discrete_factor"],
#@        "mentioned_pre_se_2010_2014_vs_2015_2021": pre_mentioned["se"],
#@        "mentioned_pre_p_2010_2014_vs_2015_2021": pre_mentioned["p"],
#@        "mentioned_pre_ci_low_2010_2014_vs_2015_2021": pre_mentioned["ci_low"],
#@        "mentioned_pre_ci_high_2010_2014_vs_2015_2021": pre_mentioned["ci_high"],
#@
#@        "mentioned_base_period": "2015-2021",
#@        "mentioned_base_event_time": -1,
#@        "mentioned_base_coef_2015_2021": 0.0,
#@        "mentioned_base_se_2015_2021": 0.0,
#@        "mentioned_base_p_2015_2021": np.nan,
#@        "mentioned_base_ci_low_2015_2021": 0.0,
#@        "mentioned_base_ci_high_2015_2021": 0.0,
#@
#@        "mentioned_post_period": "2022-2024",
#@        "mentioned_post_event_time": 1,
#@        "mentioned_post_coef_2022_2024_vs_2015_2021": post_mentioned["coef"],
#@        "mentioned_post_model_log1p_coef_2022_2024_vs_2015_2021": post_mentioned["model_coef"],
#@        "mentioned_post_average_discrete_factor": post_mentioned["average_discrete_factor"],
#@        "mentioned_post_se_2022_2024_vs_2015_2021": post_mentioned["se"],
#@        "mentioned_post_p_2022_2024_vs_2015_2021": post_mentioned["p"],
#@        "mentioned_post_ci_low_2022_2024_vs_2015_2021": post_mentioned["ci_low"],
#@        "mentioned_post_ci_high_2022_2024_vs_2015_2021": post_mentioned["ci_high"],
#@    }
#@
#@
#@def make_final_summary_row_2015(ycol, treat_col, pre_coef_df, post_coef_df, pre_panel, post_panel):
#@    pre_origin = extract_term_or_nan(pre_coef_df, "PRE_origin")
#@    pre_mentioned = extract_term_or_nan(pre_coef_df, "PRE_mentioned")
#@    post_origin = extract_term_or_nan(post_coef_df, "POST_origin")
#@    post_mentioned = extract_term_or_nan(post_coef_df, "POST_mentioned")
#@
#@    origin_parallel = "pass" if pd.notna(pre_origin["p"]) and pre_origin["p"] >= 0.05 else "fail"
#@    mentioned_parallel = "pass" if pd.notna(pre_mentioned["p"]) and pre_mentioned["p"] >= 0.05 else "fail"
#@    overall_parallel = "pass_both_sides" if origin_parallel == "pass" and mentioned_parallel == "pass" else "fail_at_least_one_side"
#@
#@    return {
#@        "outcome": ycol,
#@        "treatment": treat_col,
#@        "model_treatment_scale": "log(1 + first-time asylum applicants per 1,000 population)",
#@        "reported_effect_scale": "sample-average discrete effect of +1 applicant per 1,000 residents",
#@        "estimator": "two-period pair and time fixed-effects DID",
#@        "covariance": "pair-clustered standard errors",
#@
#@        "n_pairs_pre_total": pre_panel["pair"].nunique(),
#@        "n_obs_pre": len(pre_panel),
#@        "n_pairs_post_common": post_panel["pair"].nunique(),
#@        "n_obs_post": len(post_panel),
#@
#@        "origin_parallel": origin_parallel,
#@        "mentioned_parallel": mentioned_parallel,
#@        "overall_parallel": overall_parallel,
#@
#@        "origin_pre_period": "2010-2012",
#@        "origin_pre_event_time": -2,
#@        "origin_pre_coef_2010_2012_vs_2013_2014": pre_origin["coef"],
#@        "origin_pre_model_log1p_coef_2010_2012_vs_2013_2014": pre_origin["model_coef"],
#@        "origin_pre_average_discrete_factor": pre_origin["average_discrete_factor"],
#@        "origin_pre_se_2010_2012_vs_2013_2014": pre_origin["se"],
#@        "origin_pre_p_2010_2012_vs_2013_2014": pre_origin["p"],
#@        "origin_pre_ci_low_2010_2012_vs_2013_2014": pre_origin["ci_low"],
#@        "origin_pre_ci_high_2010_2012_vs_2013_2014": pre_origin["ci_high"],
#@
#@        "origin_base_period": "2013-2014",
#@        "origin_base_event_time": -1,
#@        "origin_base_coef_2013_2014": 0.0,
#@        "origin_base_se_2013_2014": 0.0,
#@        "origin_base_p_2013_2014": np.nan,
#@        "origin_base_ci_low_2013_2014": 0.0,
#@        "origin_base_ci_high_2013_2014": 0.0,
#@
#@        "origin_post_period": "2015-2021",
#@        "origin_post_event_time": 1,
#@        "origin_post_coef_2015_2021_vs_2013_2014": post_origin["coef"],
#@        "origin_post_model_log1p_coef_2015_2021_vs_2013_2014": post_origin["model_coef"],
#@        "origin_post_average_discrete_factor": post_origin["average_discrete_factor"],
#@        "origin_post_se_2015_2021_vs_2013_2014": post_origin["se"],
#@        "origin_post_p_2015_2021_vs_2013_2014": post_origin["p"],
#@        "origin_post_ci_low_2015_2021_vs_2013_2014": post_origin["ci_low"],
#@        "origin_post_ci_high_2015_2021_vs_2013_2014": post_origin["ci_high"],
#@
#@        "mentioned_pre_period": "2010-2012",
#@        "mentioned_pre_event_time": -2,
#@        "mentioned_pre_coef_2010_2012_vs_2013_2014": pre_mentioned["coef"],
#@        "mentioned_pre_model_log1p_coef_2010_2012_vs_2013_2014": pre_mentioned["model_coef"],
#@        "mentioned_pre_average_discrete_factor": pre_mentioned["average_discrete_factor"],
#@        "mentioned_pre_se_2010_2012_vs_2013_2014": pre_mentioned["se"],
#@        "mentioned_pre_p_2010_2012_vs_2013_2014": pre_mentioned["p"],
#@        "mentioned_pre_ci_low_2010_2012_vs_2013_2014": pre_mentioned["ci_low"],
#@        "mentioned_pre_ci_high_2010_2012_vs_2013_2014": pre_mentioned["ci_high"],
#@
#@        "mentioned_base_period": "2013-2014",
#@        "mentioned_base_event_time": -1,
#@        "mentioned_base_coef_2013_2014": 0.0,
#@        "mentioned_base_se_2013_2014": 0.0,
#@        "mentioned_base_p_2013_2014": np.nan,
#@        "mentioned_base_ci_low_2013_2014": 0.0,
#@        "mentioned_base_ci_high_2013_2014": 0.0,
#@
#@        "mentioned_post_period": "2015-2021",
#@        "mentioned_post_event_time": 1,
#@        "mentioned_post_coef_2015_2021_vs_2013_2014": post_mentioned["coef"],
#@        "mentioned_post_model_log1p_coef_2015_2021_vs_2013_2014": post_mentioned["model_coef"],
#@        "mentioned_post_average_discrete_factor": post_mentioned["average_discrete_factor"],
#@        "mentioned_post_se_2015_2021_vs_2013_2014": post_mentioned["se"],
#@        "mentioned_post_p_2015_2021_vs_2013_2014": post_mentioned["p"],
#@        "mentioned_post_ci_low_2015_2021_vs_2013_2014": post_mentioned["ci_low"],
#@        "mentioned_post_ci_high_2015_2021_vs_2013_2014": post_mentioned["ci_high"],
#@    }
#@
#@
#@# ============================================================
#@# RUN ONE SPEC - 2022 DID VERSION
#@# ============================================================
#@def run_one_spec_2022(combined_path, yearly_path, asylum_path, ycol, treat_col,
#@                      min_count=30, include_self=False, use_weights=False,
#@                      sample_spec=SAMPLE_FULL, omitted_country=None,
#@                      return_panels=False):
#@
#@    # A. PRE-TREND: 2010-2014 vs 2015-2021
#@    pre_panel = build_pre_panel_2022_from_yearly(
#@        yearly_path=yearly_path,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@    pre_panel = attach_treatment_two_sides(
#@        panel=pre_panel,
#@        asylum_path=asylum_path,
#@        treat_col=treat_col,
#@        treat_year=2022
#@    )
#@    pre_panel = pre_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
#@    pre_panel, _ = keep_common_pairs(pre_panel, ["2010-2014", "2015-2021"])
#@    pre_before_filter = pre_panel.copy(deep=True)
#@    pre_panel = apply_sensitivity_filter(
#@        pre_panel, sample_spec=sample_spec, omitted_country=omitted_country
#@    )
#@    pre_panel = build_pretrend_vars_2022(pre_panel)
#@
#@    pre_res, _ = run_panel_ols(
#@        pre_panel,
#@        ["PRE_origin", "PRE_mentioned"],
#@        "pair",
#@        use_weights
#@    )
#@    pre_coef = coef_table_from_res(pre_res)
#@    pre_coef = convert_coef_table_to_average_discrete_effect(
#@        pre_coef,
#@        pre_panel,
#@        {
#@            "PRE_origin": "raw_treat_origin",
#@            "PRE_mentioned": "raw_treat_mentioned",
#@        },
#@    )
#@
#@    # B. POST-DID: 2015-2021 vs 2022-2024
#@    post_panel = read_panel_from_period_file(
#@        combined_path=combined_path,
#@        sheets=["2015-2021", "2022-2024"],
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@    post_panel, _ = keep_common_pairs(post_panel, ["2015-2021", "2022-2024"])
#@    post_panel = attach_treatment_two_sides(
#@        panel=post_panel,
#@        asylum_path=asylum_path,
#@        treat_col=treat_col,
#@        treat_year=2022
#@    )
#@    post_panel = post_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
#@    post_before_filter = post_panel.copy(deep=True)
#@    post_panel = apply_sensitivity_filter(
#@        post_panel, sample_spec=sample_spec, omitted_country=omitted_country
#@    )
#@    post_panel = build_post_did_vars_2022(post_panel)
#@
#@    post_res, _ = run_panel_ols(
#@        post_panel,
#@        ["POST_origin", "POST_mentioned"],
#@        "pair",
#@        use_weights
#@    )
#@    post_coef = coef_table_from_res(post_res)
#@    post_coef = convert_coef_table_to_average_discrete_effect(
#@        post_coef,
#@        post_panel,
#@        {
#@            "POST_origin": "raw_treat_origin",
#@            "POST_mentioned": "raw_treat_mentioned",
#@        },
#@    )
#@
#@    summary = make_final_summary_row_2022(
#@        ycol=ycol,
#@        treat_col=treat_col,
#@        pre_coef_df=pre_coef,
#@        post_coef_df=post_coef,
#@        pre_panel=pre_panel,
#@        post_panel=post_panel
#@    )
#@    summary = add_sensitivity_metadata(
#@        summary=summary,
#@        crisis="2022",
#@        sample_spec=sample_spec,
#@        omitted_country=omitted_country,
#@        pre_before=pre_before_filter,
#@        post_before=post_before_filter,
#@        pre_after=pre_panel,
#@        post_after=post_panel,
#@    )
#@    if return_panels:
#@        return summary, pre_panel.copy(deep=True), post_panel.copy(deep=True)
#@    return summary
#@
#@
#@# ============================================================
#@# RUN ONE SPEC - 2015 CRISIS VERSION
#@# ============================================================
#@def run_one_spec_2015(combined_path, yearly_path, asylum_path, ycol, treat_col,
#@                      min_count=20, include_self=False, use_weights=False,
#@                      sample_spec=SAMPLE_FULL, omitted_country=None,
#@                      return_panels=False):
#@
#@    # A. PRE-TREND: 2010-2012 vs 2013-2014
#@    pre_panel = build_pre_panel_2015_from_yearly(
#@        yearly_path=yearly_path,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@    pre_panel = attach_dyadic_treat_k_2015(pre_panel, asylum_path, treat_col=treat_col)
#@    pre_panel = pre_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
#@    pre_before_filter = pre_panel.copy(deep=True)
#@    pre_panel = apply_sensitivity_filter(
#@        pre_panel, sample_spec=sample_spec, omitted_country=omitted_country
#@    )
#@    pre_panel = build_pretrend_vars_2015(pre_panel)
#@
#@    pre_res, _ = run_panel_ols(
#@        pre_panel,
#@        ["PRE_origin", "PRE_mentioned"],
#@        "pair",
#@        use_weights
#@    )
#@    pre_coef = coef_table_from_res(pre_res)
#@    pre_coef = convert_coef_table_to_average_discrete_effect(
#@        pre_coef,
#@        pre_panel,
#@        {
#@            "PRE_origin": "raw_treat_origin_2015",
#@            "PRE_mentioned": "raw_treat_mentioned_2015",
#@        },
#@    )
#@
#@    # B. DID POST: 2013-2014 vs 2015-2021
#@    baseline_panel = build_pre_panel_2015_from_yearly(
#@        yearly_path=yearly_path,
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@    baseline_panel = baseline_panel[baseline_panel["period"] == "2013-2014"].copy()
#@
#@    post_period_panel = read_panel_from_period_file(
#@        combined_path=combined_path,
#@        sheets=["2015-2021"],
#@        ycol=ycol,
#@        min_count=min_count,
#@        include_self=include_self
#@    )
#@
#@    post_panel = pd.concat([
#@        baseline_panel[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den"]],
#@        post_period_panel[["pair", "origin", "mentioned", "period", "Y", "post_count"]]
#@    ], ignore_index=True)
#@
#@    post_panel, _ = keep_common_pairs(post_panel, ["2013-2014", "2015-2021"])
#@    post_panel = attach_dyadic_treat_k_2015(post_panel, asylum_path, treat_col=treat_col)
#@    post_panel = post_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
#@    post_before_filter = post_panel.copy(deep=True)
#@    post_panel = apply_sensitivity_filter(
#@        post_panel, sample_spec=sample_spec, omitted_country=omitted_country
#@    )
#@    post_panel = build_post_did_vars_2015(post_panel)
#@
#@    post_res, _ = run_panel_ols(
#@        post_panel,
#@        ["POST_origin", "POST_mentioned"],
#@        "pair",
#@        use_weights
#@    )
#@    post_coef = coef_table_from_res(post_res)
#@    post_coef = convert_coef_table_to_average_discrete_effect(
#@        post_coef,
#@        post_panel,
#@        {
#@            "POST_origin": "raw_treat_origin_2015",
#@            "POST_mentioned": "raw_treat_mentioned_2015",
#@        },
#@    )
#@
#@    summary = make_final_summary_row_2015(
#@        ycol=ycol,
#@        treat_col=treat_col,
#@        pre_coef_df=pre_coef,
#@        post_coef_df=post_coef,
#@        pre_panel=pre_panel,
#@        post_panel=post_panel
#@    )
#@    summary = add_sensitivity_metadata(
#@        summary=summary,
#@        crisis="2015",
#@        sample_spec=sample_spec,
#@        omitted_country=omitted_country,
#@        pre_before=pre_before_filter,
#@        post_before=post_before_filter,
#@        pre_after=pre_panel,
#@        post_after=post_panel,
#@    )
#@    if return_panels:
#@        return summary, pre_panel.copy(deep=True), post_panel.copy(deep=True)
#@    return summary
#@
#@
#@# ============================================================
#@# PERIOD LABELS FOR FIGURES
#@# ============================================================
#@def format_period_labels_type1(periods):
#@    mapping = {
#@        "2010-2014": "2010-2014\n(pre 1)",
#@        "2015-2021": "2015-2021\n(baseline)",
#@        "2022-2024": "2022-2024\n(post 1)",
#@    }
#@    return [mapping.get(p, str(p)) for p in periods]
#@
#@
#@def format_period_labels_type2(periods):
#@    mapping = {
#@        "2010-2012": "2010-2012\n(pre 1)",
#@        "2013-2014": "2013-2014\n(baseline)",
#@        "2015-2021": "2015-2021\n(post 1)",
#@    }
#@    return [mapping.get(p, str(p)) for p in periods]
#@
#@
#@# ============================================================
#@# GET SERIES - DATASET 1: 2022 DID
#@# ============================================================
#@def get_series_type1(row, side):
#@    x = unified_x()
#@
#@    if side == "origin":
#@        periods = [
#@            row["origin_pre_period"],
#@            row["origin_base_period"],
#@            row["origin_post_period"]
#@        ]
#@
#@        y = [
#@            row["origin_pre_coef_2010_2014_vs_2015_2021"],
#@            row["origin_base_coef_2015_2021"],
#@            row["origin_post_coef_2022_2024_vs_2015_2021"]
#@        ]
#@
#@        ci_low = [
#@            row["origin_pre_ci_low_2010_2014_vs_2015_2021"],
#@            row["origin_base_ci_low_2015_2021"],
#@            row["origin_post_ci_low_2022_2024_vs_2015_2021"]
#@        ]
#@
#@        ci_high = [
#@            row["origin_pre_ci_high_2010_2014_vs_2015_2021"],
#@            row["origin_base_ci_high_2015_2021"],
#@            row["origin_post_ci_high_2022_2024_vs_2015_2021"]
#@        ]
#@
#@    else:
#@        periods = [
#@            row["mentioned_pre_period"],
#@            row["mentioned_base_period"],
#@            row["mentioned_post_period"]
#@        ]
#@
#@        y = [
#@            row["mentioned_pre_coef_2010_2014_vs_2015_2021"],
#@            row["mentioned_base_coef_2015_2021"],
#@            row["mentioned_post_coef_2022_2024_vs_2015_2021"]
#@        ]
#@
#@        ci_low = [
#@            row["mentioned_pre_ci_low_2010_2014_vs_2015_2021"],
#@            row["mentioned_base_ci_low_2015_2021"],
#@            row["mentioned_post_ci_low_2022_2024_vs_2015_2021"]
#@        ]
#@
#@        ci_high = [
#@            row["mentioned_pre_ci_high_2010_2014_vs_2015_2021"],
#@            row["mentioned_base_ci_high_2015_2021"],
#@            row["mentioned_post_ci_high_2022_2024_vs_2015_2021"]
#@        ]
#@
#@    yerr_lower = [
#@        y[i] - ci_low[i] if pd.notna(y[i]) and pd.notna(ci_low[i]) else np.nan
#@        for i in range(3)
#@    ]
#@
#@    yerr_upper = [
#@        ci_high[i] - y[i] if pd.notna(y[i]) and pd.notna(ci_high[i]) else np.nan
#@        for i in range(3)
#@    ]
#@
#@    return x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high
#@
#@
#@# ============================================================
#@# GET SERIES - DATASET 2: 2015 CRISIS
#@# ============================================================
#@def get_series_type2(row, side):
#@    x = unified_x()
#@
#@    if side == "origin":
#@        periods = [
#@            row["origin_pre_period"],
#@            row["origin_base_period"],
#@            row["origin_post_period"]
#@        ]
#@
#@        y = [
#@            row["origin_pre_coef_2010_2012_vs_2013_2014"],
#@            row["origin_base_coef_2013_2014"],
#@            row["origin_post_coef_2015_2021_vs_2013_2014"]
#@        ]
#@
#@        ci_low = [
#@            row["origin_pre_ci_low_2010_2012_vs_2013_2014"],
#@            row["origin_base_ci_low_2013_2014"],
#@            row["origin_post_ci_low_2015_2021_vs_2013_2014"]
#@        ]
#@
#@        ci_high = [
#@            row["origin_pre_ci_high_2010_2012_vs_2013_2014"],
#@            row["origin_base_ci_high_2013_2014"],
#@            row["origin_post_ci_high_2015_2021_vs_2013_2014"]
#@        ]
#@
#@    else:
#@        periods = [
#@            row["mentioned_pre_period"],
#@            row["mentioned_base_period"],
#@            row["mentioned_post_period"]
#@        ]
#@
#@        y = [
#@            row["mentioned_pre_coef_2010_2012_vs_2013_2014"],
#@            row["mentioned_base_coef_2013_2014"],
#@            row["mentioned_post_coef_2015_2021_vs_2013_2014"]
#@        ]
#@
#@        ci_low = [
#@            row["mentioned_pre_ci_low_2010_2012_vs_2013_2014"],
#@            row["mentioned_base_ci_low_2013_2014"],
#@            row["mentioned_post_ci_low_2015_2021_vs_2013_2014"]
#@        ]
#@
#@        ci_high = [
#@            row["mentioned_pre_ci_high_2010_2012_vs_2013_2014"],
#@            row["mentioned_base_ci_high_2013_2014"],
#@            row["mentioned_post_ci_high_2015_2021_vs_2013_2014"]
#@        ]
#@
#@    yerr_lower = [
#@        y[i] - ci_low[i] if pd.notna(y[i]) and pd.notna(ci_low[i]) else np.nan
#@        for i in range(3)
#@    ]
#@
#@    yerr_upper = [
#@        ci_high[i] - y[i] if pd.notna(y[i]) and pd.notna(ci_high[i]) else np.nan
#@        for i in range(3)
#@    ]
#@
#@    return x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high
#@
#@
#@# ============================================================
#@# SHARED Y-AXIS FOR THE TWO SPECIAL FIGURES
#@# ============================================================
#@def build_shared_special_ylim(df1, df2):
#@    all_ci_low = []
#@    all_ci_high = []
#@
#@    sub1 = df1[
#@        (df1["outcome"].astype(str) == SPECIAL_OUTCOME)
#@        & (df1["treatment"].astype(str).isin(SPECIAL_TREATMENTS))
#@    ]
#@
#@    if len(sub1) > 0:
#@        row1 = sub1.iloc[0]
#@        _, _, _, _, _, ci_low1, ci_high1 = get_series_type1(row1, SPECIAL_SIDE)
#@        all_ci_low.extend(ci_low1)
#@        all_ci_high.extend(ci_high1)
#@
#@    sub2 = df2[
#@        (df2["outcome"].astype(str) == SPECIAL_OUTCOME)
#@        & (df2["treatment"].astype(str).isin(SPECIAL_TREATMENTS))
#@    ]
#@
#@    if len(sub2) > 0:
#@        row2 = sub2.iloc[0]
#@        _, _, _, _, _, ci_low2, ci_high2 = get_series_type2(row2, SPECIAL_SIDE)
#@        all_ci_low.extend(ci_low2)
#@        all_ci_high.extend(ci_high2)
#@
#@    if len(all_ci_low) == 0 or len(all_ci_high) == 0:
#@        return None
#@
#@    return nice_ylim_raw(all_ci_low, all_ci_high)
#@
#@
#@# ============================================================
#@# DRAW SINGLE REFINED FIGURE
#@# ============================================================
#@def draw_single_coef_plot(row, side, dataset_type, prefix, output_dir, special_shared_ylim):
#@    outcome_value = row["outcome"]
#@    color = get_line_color(outcome_value)
#@
#@    if dataset_type == "2022":
#@        x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high = get_series_type1(row, side)
#@        labels = format_period_labels_type1(periods)
#@    else:
#@        x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high = get_series_type2(row, side)
#@        labels = format_period_labels_type2(periods)
#@
#@    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
#@
#@    # Draw only the estimated pre- and post-period coefficients.
#@    # The baseline is the omitted reference category fixed at zero, so keep
#@    # its x-axis label but do not draw a coefficient marker at the baseline.
#@    for i in (0, 2):
#@        draw_coef_point(
#@            ax=ax,
#@            x_value=x[i],
#@            y_value=y[i],
#@            yerr_lower=yerr_lower[i],
#@            yerr_upper=yerr_upper[i],
#@            ci_low=ci_low[i],
#@            ci_high=ci_high[i],
#@            color=color,
#@            zorder=4
#@        )
#@
#@    is_special = use_special_tick_labels(row, side, prefix)
#@
#@    if is_special and special_shared_ylim is not None:
#@        ymin, ymax = special_shared_ylim
#@    else:
#@        ymin, ymax = nice_ylim_raw(ci_low, ci_high)
#@
#@    ax.set_ylim(ymin, ymax)
#@    style_axis(ax)
#@
#@    if is_special and RELABEL_SPECIAL_YTICKS:
#@        relabel_top_bottom_yticks(
#@            ax,
#@            bottom_label=SPECIAL_BOTTOM_LABEL,
#@            top_label=SPECIAL_TOP_LABEL
#@        )
#@
#@    ax.set_xticks(x)
#@    ax.set_xticklabels(labels)
#@
#@    if SHOW_YLABEL:
#@        ax.set_ylabel(
#@            "Average discrete effect\n(+1 applicant per 1,000)",
#@            fontsize=LABEL_SIZE,
#@        )
#@
#@    if SHOW_POST_LABEL and pd.notna(y[2]):
#@        label = format_coef_label(y[2], ci_low[2], ci_high[2])
#@
#@        ax.annotate(
#@            label,
#@            xy=(x[2], y[2]),
#@            xytext=(9, 0),
#@            textcoords="offset points",
#@            ha="left",
#@            va="center",
#@            fontsize=ANNOT_SIZE,
#@            color=color,
#@            fontweight="bold",
#@            zorder=6
#@        )
#@
#@    fname = make_figure_filename(
#@        prefix=prefix,
#@        outcome=row["outcome"],
#@        treatment=row["treatment"],
#@        side=side
#@    )
#@
#@    plt.tight_layout()
#@
#@    plt.savefig(
#@        os.path.join(output_dir, fname),
#@        dpi=600,
#@        bbox_inches="tight"
#@    )
#@
#@    plt.close()
#@
#@
#@def draw_all_refined_figures(results_2022_csv, results_2015_csv, figure_dir):
#@    safe_mkdir(figure_dir)
#@
#@    df1 = pd.read_csv(results_2022_csv)
#@    df2 = pd.read_csv(results_2015_csv)
#@
#@    special_shared_ylim = build_shared_special_ylim(df1, df2)
#@
#@    # 2022 crisis figures
#@    for _, row in df1.iterrows():
#@        draw_single_coef_plot(row, "origin", "2022", "2022", figure_dir, special_shared_ylim)
#@        draw_single_coef_plot(row, "mentioned", "2022", "2022", figure_dir, special_shared_ylim)
#@
#@    # 2015 crisis figures
#@    for _, row in df2.iterrows():
#@        draw_single_coef_plot(row, "origin", "crisis2015", "crisis2015", figure_dir, special_shared_ylim)
#@        draw_single_coef_plot(row, "mentioned", "crisis2015", "crisis2015", figure_dir, special_shared_ylim)
#@
#@
#@# ============================================================
#@# GEOPOLITICAL SENSITIVITY ANALYSES
#@# ============================================================
#@def _sensitivity_error_row(crisis, ycol, treat_col, sample_spec,
#@                           omitted_country, exc):
#@    return {
#@        "crisis": str(crisis),
#@        "outcome": str(ycol),
#@        "treatment": str(treat_col),
#@        "sample_spec": str(sample_spec),
#@        "omitted_country": (
#@            "" if omitted_country is None else clean_country_name(omitted_country)
#@        ),
#@        "status": "error",
#@        "error_message": f"{type(exc).__name__}: {exc}",
#@    }
#@
#@
#@def _run_sensitivity_scenario(run_func, crisis, ycol, treat_col,
#@                              min_count, sample_spec, omitted_country=None,
#@                              return_panels=False):
#@    kwargs = dict(
#@        combined_path=combined_path,
#@        yearly_path=yearly_path,
#@        asylum_path=asylum_path,
#@        ycol=ycol,
#@        treat_col=treat_col,
#@        min_count=min_count,
#@        include_self=INCLUDE_SELF,
#@        use_weights=USE_WEIGHTS,
#@        sample_spec=sample_spec,
#@        omitted_country=omitted_country,
#@        return_panels=return_panels,
#@    )
#@    return run_func(**kwargs)
#@
#@
#@def run_geopolitical_sensitivity_for_crisis(crisis: str):
#@    """Run the full benchmark, S1, and leave-one-country-out specifications."""
#@    if str(crisis) == "2022":
#@        outcomes = DID2022_OUTCOME_SPECS
#@        treatments = DID2022_TREAT_SPECS
#@        min_count = DID2022_MIN_COUNT
#@        run_func = run_one_spec_2022
#@    elif str(crisis) == "2015":
#@        outcomes = CRISIS2015_OUTCOME_SPECS
#@        treatments = CRISIS2015_TREAT_SPECS
#@        min_count = CRISIS2015_MIN_COUNT
#@        run_func = run_one_spec_2015
#@    else:
#@        raise ValueError(f"Unknown crisis: {crisis}")
#@
#@    rows = []
#@
#@    for outcome in outcomes:
#@        ycol = outcome["ycol"]
#@        for treat in treatments:
#@            treat_col = treat["treat_col"]
#@            print(
#@                f"[SENSITIVITY {crisis}] Preparing full sample: "
#@                f"Y={ycol}, Treat={treat_col}"
#@            )
#@
#@            # The full model must succeed.  It defines the actual country
#@            # universe for the leave-one-country-out loop.
#@            full_row, full_pre_panel, full_post_panel = _run_sensitivity_scenario(
#@                run_func=run_func,
#@                crisis=crisis,
#@                ycol=ycol,
#@                treat_col=treat_col,
#@                min_count=min_count,
#@                sample_spec=SAMPLE_FULL,
#@                omitted_country=None,
#@                return_panels=True,
#@            )
#@            rows.append(full_row)
#@
#@            countries = analysis_country_universe(full_pre_panel, full_post_panel)
#@
#@            # S1 is the fixed, prespecified exclusion of the two direct
#@            # Russia-Ukraine dyads.
#@            for sample_spec in [SAMPLE_S1]:
#@                print(
#@                    f"[SENSITIVITY {crisis}] {sample_spec}: "
#@                    f"Y={ycol}, Treat={treat_col}"
#@                )
#@                try:
#@                    row = _run_sensitivity_scenario(
#@                        run_func=run_func,
#@                        crisis=crisis,
#@                        ycol=ycol,
#@                        treat_col=treat_col,
#@                        min_count=min_count,
#@                        sample_spec=sample_spec,
#@                        omitted_country=None,
#@                        return_panels=False,
#@                    )
#@                except Exception as exc:
#@                    row = _sensitivity_error_row(
#@                        crisis, ycol, treat_col, sample_spec, None, exc
#@                    )
#@                rows.append(row)
#@
#@            # S3 omits every country that actually appears in this model's
#@            # full pre/post analysis sample.  No country is selected according
#@            # to the resulting sign or p value.
#@            for omitted_country in countries:
#@                print(
#@                    f"[SENSITIVITY {crisis}] Leaving out {omitted_country}: "
#@                    f"Y={ycol}, Treat={treat_col}"
#@                )
#@                try:
#@                    row = _run_sensitivity_scenario(
#@                        run_func=run_func,
#@                        crisis=crisis,
#@                        ycol=ycol,
#@                        treat_col=treat_col,
#@                        min_count=min_count,
#@                        sample_spec=SAMPLE_S3,
#@                        omitted_country=omitted_country,
#@                        return_panels=False,
#@                    )
#@                except Exception as exc:
#@                    row = _sensitivity_error_row(
#@                        crisis, ycol, treat_col, SAMPLE_S3,
#@                        omitted_country, exc
#@                    )
#@                rows.append(row)
#@
#@    return pd.DataFrame(rows)
#@
#@
#@def _post_result_columns(crisis: str, side: str):
#@    if str(crisis) == "2022":
#@        suffix = "2022_2024_vs_2015_2021"
#@    elif str(crisis) == "2015":
#@        suffix = "2015_2021_vs_2013_2014"
#@    else:
#@        raise ValueError(f"Unknown crisis: {crisis}")
#@
#@    return {
#@        "coef": f"{side}_post_coef_{suffix}",
#@        "se": f"{side}_post_se_{suffix}",
#@        "p": f"{side}_post_p_{suffix}",
#@        "ci_low": f"{side}_post_ci_low_{suffix}",
#@        "ci_high": f"{side}_post_ci_high_{suffix}",
#@        "model_coef": f"{side}_post_model_log1p_coef_{suffix}",
#@        "factor": f"{side}_post_average_discrete_factor",
#@    }
#@
#@
#@def sensitivity_wide_to_post_long(wide_df: pd.DataFrame):
#@    """Create one transparent row per post-DID side and sample scenario."""
#@    metadata_cols = [
#@        "crisis", "outcome", "treatment", "sample_spec",
#@        "omitted_country", "status", "error_message",
#@        "n_rows_pre_before_filter", "n_rows_pre_after_filter",
#@        "n_rows_post_before_filter", "n_rows_post_after_filter",
#@        "n_pairs_pre_before_filter", "n_pairs_pre_after_filter",
#@        "n_pairs_post_before_filter", "n_pairs_post_after_filter",
#@        "n_unique_countries_pre_after_filter",
#@        "n_unique_countries_post_after_filter",
#@        "n_rows_pre_removed", "n_rows_post_removed",
#@        "n_pairs_pre_removed", "n_pairs_post_removed",
#@        "covariance", "estimator", "model_treatment_scale",
#@        "reported_effect_scale",
#@    ]
#@
#@    rows = []
#@    for _, source_row in wide_df.iterrows():
#@        crisis = str(source_row.get("crisis", ""))
#@        for side in ["origin", "mentioned"]:
#@            cols = _post_result_columns(crisis, side)
#@            row = {c: source_row.get(c, np.nan) for c in metadata_cols}
#@            row.update({
#@                "side": side,
#@                "coef": source_row.get(cols["coef"], np.nan),
#@                "se": source_row.get(cols["se"], np.nan),
#@                "p": source_row.get(cols["p"], np.nan),
#@                "ci_low": source_row.get(cols["ci_low"], np.nan),
#@                "ci_high": source_row.get(cols["ci_high"], np.nan),
#@                "model_log1p_coef": source_row.get(cols["model_coef"], np.nan),
#@                "average_discrete_factor": source_row.get(cols["factor"], np.nan),
#@            })
#@            rows.append(row)
#@
#@    out = pd.DataFrame(rows)
#@    numeric_cols = [
#@        "coef", "se", "p", "ci_low", "ci_high", "model_log1p_coef",
#@        "average_discrete_factor"
#@    ]
#@    for col in numeric_cols:
#@        out[col] = pd.to_numeric(out[col], errors="coerce")
#@
#@    key_cols = ["crisis", "outcome", "treatment", "side"]
#@    full = (
#@        out[out["sample_spec"] == SAMPLE_FULL][key_cols + ["coef", "p", "ci_low", "ci_high"]]
#@        .rename(columns={
#@            "coef": "full_sample_coef",
#@            "p": "full_sample_p",
#@            "ci_low": "full_sample_ci_low",
#@            "ci_high": "full_sample_ci_high",
#@        })
#@        .drop_duplicates(key_cols)
#@    )
#@    out = out.merge(full, on=key_cols, how="left")
#@    out["coef_change_from_full"] = out["coef"] - out["full_sample_coef"]
#@    out["abs_coef_change_from_full"] = out["coef_change_from_full"].abs()
#@    out["relative_coef_change_from_full"] = np.where(
#@        out["full_sample_coef"].abs() > 0,
#@        out["coef_change_from_full"] / out["full_sample_coef"].abs(),
#@        np.nan,
#@    )
#@    out["same_sign_as_full"] = np.where(
#@        out["coef"].notna() & out["full_sample_coef"].notna()
#@        & (out["coef"] != 0) & (out["full_sample_coef"] != 0),
#@        np.sign(out["coef"]) == np.sign(out["full_sample_coef"]),
#@        np.nan,
#@    )
#@    out["ci_excludes_zero"] = (
#@        ((out["ci_low"] > 0) & (out["ci_high"] > 0))
#@        | ((out["ci_low"] < 0) & (out["ci_high"] < 0))
#@    )
#@    return out
#@
#@
#@def build_leave_one_country_summary(post_long: pd.DataFrame):
#@    loo = post_long[
#@        (post_long["sample_spec"] == SAMPLE_S3)
#@        & (post_long["status"] == "ok")
#@        & post_long["coef"].notna()
#@    ].copy()
#@
#@    rows = []
#@    group_cols = ["crisis", "outcome", "treatment", "side"]
#@    for keys, group in loo.groupby(group_cols, dropna=False):
#@        group = group.sort_values("abs_coef_change_from_full", ascending=False)
#@        top = group.iloc[0]
#@        rows.append({
#@            "crisis": keys[0],
#@            "outcome": keys[1],
#@            "treatment": keys[2],
#@            "side": keys[3],
#@            "full_sample_coef": top["full_sample_coef"],
#@            "full_sample_p": top["full_sample_p"],
#@            "n_successful_leave_one_out_models": int(len(group)),
#@            "most_influential_omitted_country": top["omitted_country"],
#@            "largest_absolute_coef_change": top["abs_coef_change_from_full"],
#@            "coef_when_most_influential_country_omitted": top["coef"],
#@            "minimum_leave_one_out_coef": group["coef"].min(),
#@            "maximum_leave_one_out_coef": group["coef"].max(),
#@            "n_sign_reversals": int((group["same_sign_as_full"] == False).sum()),
#@            "n_ci_excluding_zero": int(group["ci_excludes_zero"].sum()),
#@        })
#@    return pd.DataFrame(rows)
#@
#@
#@def _save_sensitivity_figure(fig, path_without_extension):
#@    fig.savefig(path_without_extension + ".png", dpi=600, bbox_inches="tight")
#@    if SAVE_PDF:
#@        fig.savefig(path_without_extension + ".pdf", bbox_inches="tight")
#@    plt.close(fig)
#@
#@
#@def draw_fixed_exclusion_sensitivity(group: pd.DataFrame, output_dir: str):
#@    order = [SAMPLE_FULL, SAMPLE_S1]
#@    label_map = {
#@        SAMPLE_FULL: "Full sample",
#@        SAMPLE_S1: "Exclude Russia-Ukraine\ndirect dyads",
#@    }
#@    plot_df = group[group["sample_spec"].isin(order)].copy()
#@    plot_df["sample_order"] = plot_df["sample_spec"].map(
#@        {name: i for i, name in enumerate(order)}
#@    )
#@    plot_df = plot_df.sort_values("sample_order")
#@    plot_df = plot_df[plot_df["coef"].notna()].copy()
#@    if plot_df.empty:
#@        return
#@
#@    y = np.arange(len(plot_df))
#@    coef = plot_df["coef"].to_numpy(dtype=float)
#@    lo = plot_df["ci_low"].to_numpy(dtype=float)
#@    hi = plot_df["ci_high"].to_numpy(dtype=float)
#@    xerr = np.vstack([
#@        np.maximum(coef - lo, 0),
#@        np.maximum(hi - coef, 0),
#@    ])
#@    color = get_line_color(plot_df.iloc[0]["outcome"])
#@
#@    fig, ax = plt.subplots(figsize=(7.6, 4.6))
#@    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.5, zorder=1)
#@    ax.errorbar(
#@        coef, y, xerr=xerr, fmt="o", color=color, ecolor=color,
#@        elinewidth=2.0, capsize=4.5, markersize=7.5, zorder=3
#@    )
#@    ax.set_yticks(y)
#@    ax.set_yticklabels([label_map[x] for x in plot_df["sample_spec"]])
#@    ax.invert_yaxis()
#@    ax.set_xlabel("Average discrete effect (+1 applicant per 1,000)")
#@    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
#@    for spine in ["top", "right"]:
#@        ax.spines[spine].set_visible(False)
#@    fig.tight_layout()
#@
#@    first = plot_df.iloc[0]
#@    fname = sanitize_filename(
#@        f"fixed_exclusions_{first['crisis']}_{outcome_to_filename_label(first['outcome'])}_"
#@        f"{treatment_to_filename_label(first['treatment'])}_{first['side']}"
#@    )
#@    _save_sensitivity_figure(fig, os.path.join(output_dir, fname))
#@
#@
#@def draw_leave_one_country_sensitivity(group: pd.DataFrame, output_dir: str):
#@    plot_df = group[
#@        (group["sample_spec"] == SAMPLE_S3)
#@        & (group["status"] == "ok")
#@        & group["coef"].notna()
#@    ].copy()
#@    if plot_df.empty:
#@        return
#@
#@    plot_df = plot_df.sort_values("coef", ascending=True)
#@    y = np.arange(len(plot_df))
#@    coef = plot_df["coef"].to_numpy(dtype=float)
#@    lo = plot_df["ci_low"].to_numpy(dtype=float)
#@    hi = plot_df["ci_high"].to_numpy(dtype=float)
#@    xerr = np.vstack([
#@        np.maximum(coef - lo, 0),
#@        np.maximum(hi - coef, 0),
#@    ])
#@    full_coef = float(plot_df["full_sample_coef"].iloc[0])
#@    color = get_line_color(plot_df.iloc[0]["outcome"])
#@    fig_height = max(6.0, 0.31 * len(plot_df) + 2.2)
#@
#@    fig, ax = plt.subplots(figsize=(8.2, fig_height))
#@    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.5, zorder=1)
#@    ax.axvline(
#@        full_coef, color="#333333", linewidth=1.5, linestyle="--",
#@        label="Full-sample coefficient", zorder=2
#@    )
#@    ax.errorbar(
#@        coef, y, xerr=xerr, fmt="o", color=color, ecolor=color,
#@        elinewidth=1.6, capsize=3.2, markersize=6.2, zorder=3
#@    )
#@    ax.set_yticks(y)
#@    ax.set_yticklabels([
#@        str(x).title() for x in plot_df["omitted_country"].tolist()
#@    ])
#@    ax.set_xlabel("Average discrete effect (+1 applicant per 1,000)")
#@    ax.set_ylabel("Omitted country")
#@    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
#@    ax.legend(frameon=False, loc="best", fontsize=11)
#@    for spine in ["top", "right"]:
#@        ax.spines[spine].set_visible(False)
#@    fig.tight_layout()
#@
#@    first = plot_df.iloc[0]
#@    fname = sanitize_filename(
#@        f"leave_one_country_out_{first['crisis']}_{outcome_to_filename_label(first['outcome'])}_"
#@        f"{treatment_to_filename_label(first['treatment'])}_{first['side']}"
#@    )
#@    _save_sensitivity_figure(fig, os.path.join(output_dir, fname))
#@
#@
#@def draw_all_geopolitical_sensitivity_figures(post_long: pd.DataFrame,
#@                                               figure_dir: str):
#@    safe_mkdir(figure_dir)
#@    group_cols = ["crisis", "outcome", "treatment", "side"]
#@    for _, group in post_long.groupby(group_cols, dropna=False):
#@        draw_fixed_exclusion_sensitivity(group, figure_dir)
#@        draw_leave_one_country_sensitivity(group, figure_dir)
#@
#@
#@def run_all_geopolitical_sensitivity(master_out_dir):
#@    sensitivity_dir = os.path.join(master_out_dir, "geopolitical_sensitivity")
#@    safe_mkdir(sensitivity_dir)
#@
#@    wide_frames = []
#@    output_paths = {}
#@
#@    if RUN_GEOPOLITICAL_SENSITIVITY_2022:
#@        wide_2022 = run_geopolitical_sensitivity_for_crisis("2022")
#@        path_2022 = os.path.join(
#@            sensitivity_dir, "sensitivity_results_2022crisis_wide.csv"
#@        )
#@        wide_2022.to_csv(path_2022, index=False, encoding="utf-8-sig")
#@        wide_frames.append(wide_2022)
#@        output_paths["sensitivity_2022"] = path_2022
#@
#@    if RUN_GEOPOLITICAL_SENSITIVITY_2015:
#@        wide_2015 = run_geopolitical_sensitivity_for_crisis("2015")
#@        path_2015 = os.path.join(
#@            sensitivity_dir, "sensitivity_results_2015crisis_wide.csv"
#@        )
#@        wide_2015.to_csv(path_2015, index=False, encoding="utf-8-sig")
#@        wide_frames.append(wide_2015)
#@        output_paths["sensitivity_2015"] = path_2015
#@
#@    if not wide_frames:
#@        return output_paths
#@
#@    wide_all = pd.concat(wide_frames, ignore_index=True, sort=False)
#@    post_long = sensitivity_wide_to_post_long(wide_all)
#@    post_long_path = os.path.join(
#@        sensitivity_dir, "sensitivity_post_coefficients_long.csv"
#@    )
#@    post_long.to_csv(post_long_path, index=False, encoding="utf-8-sig")
#@
#@    audit_cols = [
#@        "crisis", "outcome", "treatment", "sample_spec",
#@        "omitted_country", "status", "error_message",
#@        "n_rows_pre_before_filter", "n_rows_pre_after_filter",
#@        "n_rows_post_before_filter", "n_rows_post_after_filter",
#@        "n_pairs_pre_before_filter", "n_pairs_pre_after_filter",
#@        "n_pairs_post_before_filter", "n_pairs_post_after_filter",
#@        "n_unique_countries_pre_after_filter",
#@        "n_unique_countries_post_after_filter",
#@        "n_rows_pre_removed", "n_rows_post_removed",
#@        "n_pairs_pre_removed", "n_pairs_post_removed",
#@    ]
#@    audit_path = os.path.join(sensitivity_dir, "sensitivity_sample_audit.csv")
#@    wide_all.reindex(columns=audit_cols).to_csv(
#@        audit_path, index=False, encoding="utf-8-sig"
#@    )
#@
#@    influence_summary = build_leave_one_country_summary(post_long)
#@    influence_path = os.path.join(
#@        sensitivity_dir, "leave_one_country_out_influence_summary.csv"
#@    )
#@    influence_summary.to_csv(influence_path, index=False, encoding="utf-8-sig")
#@
#@    output_paths.update({
#@        "post_coefficients_long": post_long_path,
#@        "sample_audit": audit_path,
#@        "leave_one_country_summary": influence_path,
#@    })
#@    return output_paths
#@
#@
#@# ============================================================
#@# MAIN RUNNERS
#@# ============================================================
#@def run_2022crisis_version(master_out_dir):
#@    master_summary = []
#@
#@    for outcome in DID2022_OUTCOME_SPECS:
#@        ycol = outcome["ycol"]
#@
#@        for treat in DID2022_TREAT_SPECS:
#@            treat_col = treat["treat_col"]
#@            print(f"[2022 CRISIS] Running: Y={ycol}, Treat={treat_col}")
#@
#@            summary_row = run_one_spec_2022(
#@                combined_path=combined_path,
#@                yearly_path=yearly_path,
#@                asylum_path=asylum_path,
#@                ycol=ycol,
#@                treat_col=treat_col,
#@                min_count=DID2022_MIN_COUNT,
#@                include_self=INCLUDE_SELF,
#@                use_weights=USE_WEIGHTS
#@            )
#@            master_summary.append(summary_row)
#@
#@    out_csv = os.path.join(master_out_dir, "results_2022crisis.csv")
#@    pd.DataFrame(master_summary).to_csv(out_csv, index=False, encoding="utf-8-sig")
#@
#@    return out_csv
#@
#@
#@def run_2015crisis_version(master_out_dir):
#@    master_summary = []
#@
#@    for outcome in CRISIS2015_OUTCOME_SPECS:
#@        ycol = outcome["ycol"]
#@
#@        for treat in CRISIS2015_TREAT_SPECS:
#@            treat_col = treat["treat_col"]
#@            print(f"[2015 CRISIS] Running: Y={ycol}, Treat={treat_col}")
#@
#@            summary_row = run_one_spec_2015(
#@                combined_path=combined_path,
#@                yearly_path=yearly_path,
#@                asylum_path=asylum_path,
#@                ycol=ycol,
#@                treat_col=treat_col,
#@                min_count=CRISIS2015_MIN_COUNT,
#@                include_self=INCLUDE_SELF,
#@                use_weights=USE_WEIGHTS
#@            )
#@            master_summary.append(summary_row)
#@
#@    out_csv = os.path.join(master_out_dir, "results_2015crisis.csv")
#@    pd.DataFrame(master_summary).to_csv(out_csv, index=False, encoding="utf-8-sig")
#@
#@    return out_csv
#@
#@
#@# ============================================================
#@# MAIN
#@# ============================================================
#@def main():
#@    tag = now_tag()
#@
#@    master_out_dir = os.path.join(
#@        OUT_ROOT,
#@        f"did_sensitivity_S1_LOO_{tag}"
#@    )
#@    safe_mkdir(master_out_dir)
#@
#@    # Run only the requested robustness analyses. The full sample is retained
#@    # as the prespecified benchmark for S1 and leave-one-country-out.
#@    # No source workbook is modified; all outputs are written to a new folder.
#@    sensitivity_outputs = run_all_geopolitical_sensitivity(master_out_dir)
#@
#@    print("Finished.")
#@    print("Master folder:", master_out_dir)
#@    for label, output_path in sensitivity_outputs.items():
#@        print(f"Sensitivity {label}: {output_path}")
#@
#@
#@if __name__ == "__main__":
#@    main()
# === END EMBEDDED sentiment_dyads ===
