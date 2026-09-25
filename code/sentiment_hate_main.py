# ============================================================
# ALL DID RESULTS + SEPARATE-SCALE COEFFICIENT / DYAD-BOX FIGURES
# ============================================================
# 1) Run 2022 DID version -> output results_2022crisis.csv
# 2) Run 2015 crisis DID version -> output results_2015crisis.csv
# 3) Draw one figure with two visibly separate axes:
#    left = original pre-period, baseline, and post-DID coefficients/95% CIs;
#    right = post dyad box on an independently expanded y-axis.
#    The box is Q25-Q75, whiskers are min-max, and its center line is the
#    dyad mean, which equals the reported sample-average discrete effect.
#    No dyad dots or mean diamonds.
# ============================================================

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from linearmodels.panel import PanelOLS
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator, FixedFormatter

# ============================================================
# USER PATH SETTINGS
# ============================================================
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

combined_path = os.path.join(INPUT_DIR, "pair_results_by_period.xlsx")
yearly_path = os.path.join(INPUT_DIR, "pair_results_by_year.xlsx")
asylum_path = os.path.join(INPUT_DIR, "asylum_data.xlsx")

DEFAULT_OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "sentiment_hate_main"))
OUT_ROOT = os.environ.get("DID_OUT_ROOT", DEFAULT_OUT_ROOT)

# ============================================================
# CHECK INPUT FILES
# ============================================================
for fp in [combined_path, yearly_path, asylum_path]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Input file not found: {fp}")

os.makedirs(OUT_ROOT, exist_ok=True)

# ============================================================
# GENERAL SETTINGS
# ============================================================
INCLUDE_SELF = False
USE_WEIGHTS = False

# Treatment functional form
# Raw exposure is first-time asylum applicants per 1,000 population.
# The regression uses log(1 + exposure) to reduce leverage from the strongly
# right-skewed rate distribution and allow diminishing marginal effects.
TREATMENT_TRANSFORM = "log1p"

# Report every coefficient as the sample-average discrete effect of increasing
# the original exposure rate by 1 applicant per 1,000 residents.  The
# regression is still estimated with log(1 + rate); this is only a transparent
# post-estimation conversion back to the original, reader-friendly unit.
REPORT_AVERAGE_DISCRETE_EFFECT = True
DISCRETE_INCREASE_PER_1000 = 1.0

# 2022 DID settings
DID2022_MIN_COUNT = 30
DID2022_OUTCOME_SPECS = [
    {"ycol": "mean_sentiment_score"},
    {"ycol": "mean_hate_speech_score"},
]
DID2022_TREAT_SPECS = [
    {"treat_col": "non-Ukrainian_first_time_applicant"},
    {"treat_col": "first_time_applicants_ukrainian"},
]

# 2015 crisis DID settings
CRISIS2015_MIN_COUNT = 20
CRISIS2015_OUTCOME_SPECS = [
    {"ycol": "mean_sentiment_score"},
    {"ycol": "mean_hate_speech_score"},
]
CRISIS2015_TREAT_SPECS = [
    {"treat_col": "first_time_applicants"},
]

# ============================================================
# GLOBAL FIGURE STYLE
# ============================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# COLORS
# ============================================================
SENTIMENT_COLOR = "#1A5A1E"
HATE_COLOR = "#F36800"
ZERO_LINE_COLOR = "#A7A7A7"

# ============================================================
# FIGURE STYLE
# ============================================================
# Preserve the size and visual weight of the original coefficient figure.
# The canvas is extended only far enough to hold a compact box panel.
ORIGINAL_MAIN_WIDTH = 5.5
BOX_PANEL_WIDTH = 0.58
FIG_W = 5.5
FIG_H = 5.15

MARKER_SIZE = 70
MARKER_EDGE_WIDTH = 1.9

CAP_SIZE = 4.8
ERROR_LINE_WIDTH = 2.2
SPINE_WIDTH = 1.5

TICK_SIZE = 15
LABEL_SIZE = 17
ANNOT_SIZE = 16

# ============================================================
# OPTIONS
# ============================================================
SHOW_YLABEL = True
SHOW_POST_LABEL = False
SAVE_PDF = False
ADD_STAR_BY_CI = True

# Dyad-distribution figure settings.  These affect figures only; the DID
# regressions, reported coefficients, standard errors, p values, and CSV output
# retain the latest main specification unchanged.
BOX_WIDTH = 0.30
# Match the box, whiskers, caps, and mean line to the coefficient CI width.
BOX_LINE_WIDTH = ERROR_LINE_WIDTH

# ============================================================
# SPECIAL FIGURES
# ============================================================
SPECIAL_OUTCOME = "mean_sentiment_score"
SPECIAL_TREATMENTS = {"first_time_applicants", "non-Ukrainian_first_time_applicant"}
SPECIAL_SIDE = "mentioned"

SPECIAL_BOTTOM_LABEL = "-0.08"
SPECIAL_TOP_LABEL = "0.08"

# Do not overwrite numeric tick labels with fixed text.  This must remain
# False when coefficient scales can change (for example after log1p treatment
# transformation), otherwise the displayed axis values would be misleading.
RELABEL_SPECIAL_YTICKS = False


# ============================================================
# BASIC HELPERS
# ============================================================
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_country_name(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.lower()
    return x


def sanitize_filename(x):
    """
    Keep spaces and parentheses.
    Only replace characters that are invalid in Windows filenames.
    """
    return (
        str(x)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def extract_term_or_nan(coef_df: pd.DataFrame, term: str):
    sub = coef_df[coef_df["term"] == term].copy()
    if sub.empty:
        return {
            "term": term,
            "coef": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "model_coef": np.nan,
            "model_se": np.nan,
            "model_ci_low": np.nan,
            "model_ci_high": np.nan,
            "average_discrete_factor": np.nan,
        }

    r = sub.iloc[0]
    return {
        "term": term,
        "coef": r["coef"],
        "se": r["se"],
        "t": r["t"],
        "p": r["p"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "model_coef": r.get("model_coef", r["coef"]),
        "model_se": r.get("model_se", r["se"]),
        "model_ci_low": r.get("model_ci_low", r["ci_low"]),
        "model_ci_high": r.get("model_ci_high", r["ci_high"]),
        "average_discrete_factor": r.get("average_discrete_factor", 1.0),
    }


def coef_table_from_res(res):
    return pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.std_errors.values,
        "t": res.tstats.values,
        "p": res.pvalues.values,
        "ci_low": res.conf_int().iloc[:, 0].values,
        "ci_high": res.conf_int().iloc[:, 1].values
    })


def _pair_level_treatment_values(panel: pd.DataFrame, raw_col: str):
    """Return one raw treatment rate per dyad after checking it is constant."""
    if raw_col not in panel.columns:
        raise ValueError(f"Missing raw treatment column for reporting: {raw_col}")

    variation = panel.groupby("pair")[raw_col].nunique(dropna=False)
    if (variation > 1).any():
        bad = variation[variation > 1].index.tolist()[:5]
        raise ValueError(
            f"Treatment is not constant within dyad for {raw_col}; examples: {bad}"
        )

    values = (
        panel[["pair", raw_col]]
        .drop_duplicates("pair")[raw_col]
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
        .astype(float)
    )
    if values.empty:
        raise ValueError(f"No usable raw treatment values in {raw_col}")
    if (values < 0).any():
        raise ValueError(f"Negative raw treatment values in {raw_col}")
    return values


def average_discrete_factor(panel: pd.DataFrame, raw_col: str):
    """
    Average change in the model's treatment index when the original rate rises
    by DISCRETE_INCREASE_PER_1000 for every dyad in the estimation sample.
    """
    values = _pair_level_treatment_values(panel, raw_col)
    increase = float(DISCRETE_INCREASE_PER_1000)
    if increase <= 0:
        raise ValueError("DISCRETE_INCREASE_PER_1000 must be positive")

    if TREATMENT_TRANSFORM == "log1p":
        contrasts = np.log1p(values + increase) - np.log1p(values)
    elif TREATMENT_TRANSFORM == "raw":
        contrasts = np.full(len(values), increase, dtype=float)
    else:
        raise ValueError(f"Unknown TREATMENT_TRANSFORM: {TREATMENT_TRANSFORM}")

    return float(np.mean(contrasts))


def dyad_specific_discrete_effects(
    panel: pd.DataFrame,
    side: str,
    raw_col: str,
    model_coef: float,
):
    """
    Return one discrete-effect value per unique dyad in the estimation sample.

    With the log1p treatment, dyad ij's value is

        model_coef * [log(1 + rate_ij + 1) - log(1 + rate_ij)].

    Every retained pair is represented once.  Consequently, the arithmetic
    mean of ``dyad_discrete_effect`` uses exactly the same pair-level treatment
    distribution as ``average_discrete_factor`` and equals the reported
    sample-average discrete effect (up to floating-point precision).
    """
    if side not in {"origin", "mentioned"}:
        raise ValueError(f"Unknown side: {side}")
    if side not in panel.columns:
        raise ValueError(f"Panel is missing country column: {side}")
    if raw_col not in panel.columns:
        raise ValueError(f"Panel is missing raw treatment column: {raw_col}")

    data = panel[["pair", "origin", "mentioned", raw_col]].copy()
    data[raw_col] = pd.to_numeric(data[raw_col], errors="coerce")
    data = data.dropna(subset=["pair", "origin", "mentioned", raw_col]).copy()
    variation = data.groupby("pair")[raw_col].nunique(dropna=False)
    if (variation > 1).any():
        bad = variation[variation > 1].index.tolist()[:5]
        raise ValueError(
            f"Treatment varies within dyad for {raw_col}; examples: {bad}"
        )
    if (data[raw_col] < 0).any():
        raise ValueError(f"Negative raw treatment values in {raw_col}")

    dyad_data = (
        data.drop_duplicates("pair")
        [["pair", "origin", "mentioned", raw_col]]
        .rename(columns={raw_col: "raw_treatment_per_1000"})
        .sort_values("pair")
        .reset_index(drop=True)
    )
    dyad_data["treatment_side"] = side
    dyad_data["treatment_country"] = dyad_data[side]

    increase = float(DISCRETE_INCREASE_PER_1000)
    if TREATMENT_TRANSFORM == "log1p":
        contrast = (
            np.log1p(dyad_data["raw_treatment_per_1000"] + increase)
            - np.log1p(dyad_data["raw_treatment_per_1000"])
        )
    elif TREATMENT_TRANSFORM == "raw":
        contrast = np.full(len(dyad_data), increase, dtype=float)
    else:
        raise ValueError(f"Unknown TREATMENT_TRANSFORM: {TREATMENT_TRANSFORM}")

    dyad_data["model_scale_contrast"] = contrast
    dyad_data["dyad_discrete_effect"] = float(model_coef) * contrast
    return dyad_data


def dyad_effect_summary_fields(
    panel: pd.DataFrame,
    side: str,
    raw_col: str,
    model_coef: float,
    field_prefix: str,
):
    """Create auditable dyad JSON and summary statistics for one boxplot."""
    data = dyad_specific_discrete_effects(
        panel=panel,
        side=side,
        raw_col=raw_col,
        model_coef=model_coef,
    )
    values = data["dyad_discrete_effect"].to_numpy(dtype=float)
    records = data.to_dict(orient="records")
    return {
        f"{field_prefix}_dyad_effects_json": json.dumps(
            records, ensure_ascii=False, separators=(",", ":")
        ),
        f"{field_prefix}_dyad_n": int(len(values)),
        f"{field_prefix}_dyad_min": float(np.min(values)),
        f"{field_prefix}_dyad_q25": float(np.quantile(values, 0.25)),
        f"{field_prefix}_dyad_median": float(np.quantile(values, 0.50)),
        f"{field_prefix}_dyad_mean": float(np.mean(values)),
        f"{field_prefix}_dyad_q75": float(np.quantile(values, 0.75)),
        f"{field_prefix}_dyad_max": float(np.max(values)),
    }


def convert_coef_table_to_average_discrete_effect(
    coef_df: pd.DataFrame,
    panel: pd.DataFrame,
    term_to_raw_col: dict,
):
    """
    Convert model-scale estimates into sample-average effects of +1 applicant
    per 1,000 residents. Multiplying estimate, SE, and CI by the same positive
    factor leaves t statistics and p values unchanged.
    """
    out = coef_df.copy()
    out["model_coef"] = out["coef"]
    out["model_se"] = out["se"]
    out["model_ci_low"] = out["ci_low"]
    out["model_ci_high"] = out["ci_high"]
    out["average_discrete_factor"] = 1.0

    if not REPORT_AVERAGE_DISCRETE_EFFECT:
        return out

    for term, raw_col in term_to_raw_col.items():
        factor = average_discrete_factor(panel, raw_col)
        mask = out["term"] == term
        if not mask.any():
            continue
        out.loc[mask, "average_discrete_factor"] = factor
        for col in ["coef", "se", "ci_low", "ci_high"]:
            out.loc[mask, col] = out.loc[mask, col] * factor

    return out


def keep_common_pairs(panel: pd.DataFrame, periods):
    tmp = panel.groupby("pair")["period"].nunique()
    valid_pairs = tmp[tmp == len(periods)].index.tolist()
    return panel[panel["pair"].isin(valid_pairs)].copy(), valid_pairs


# ============================================================
# FIGURE HELPERS
# ============================================================
def get_line_color(outcome_value):
    if str(outcome_value) == "mean_sentiment_score":
        return SENTIMENT_COLOR
    return HATE_COLOR


def unified_x():
    return [0, 1, 2]


def ci_excludes_zero(ci_low, ci_high):
    if pd.isna(ci_low) or pd.isna(ci_high):
        return False
    return (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)


def format_coef_label(value, ci_low=None, ci_high=None):
    """
    Display the reported average discrete effect.
    Example: ΔY = -0.022*
    """
    if pd.isna(value):
        return ""

    star = ""
    if ADD_STAR_BY_CI and ci_excludes_zero(ci_low, ci_high):
        star = "*"

    return f"ΔY = {value:.3f}{star}"


def dynamic_formatter(x, pos):
    """
    Adapt tick precision to the magnitude of the axis.

    Important:
    The y-axis zero label is always displayed as 0.00.
    """
    if np.isclose(x, 0, atol=1e-12):
        return "0.00"

    ax_abs = abs(x)

    if ax_abs >= 1:
        return f"{x:.1f}"
    elif ax_abs >= 0.1:
        return f"{x:.2f}"
    else:
        return f"{x:.3f}"


def nice_ylim_raw(ci_low, ci_high):
    """
    Force y-axis to be symmetric around zero.
    """
    vals = [v for v in ci_low + ci_high if pd.notna(v)]

    if not vals:
        return (-0.05, 0.05)

    max_abs = max(abs(min(vals)), abs(max(vals)), 0)

    limit = max_abs * 1.22

    min_half_span = 0.03
    if limit < min_half_span:
        limit = min_half_span

    return -limit, limit


def nice_ylim_distribution(values):
    """Return a tight, non-symmetric range for the separate box-plot axis."""
    vals = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(vals) == 0:
        return (-0.05, 0.05)

    low = float(np.min(vals))
    high = float(np.max(vals))
    span = high - low
    if span <= 1e-12:
        span = max(abs(low) * 0.20, 0.01)

    padding = 0.18 * span
    return low - padding, high + padding


def style_axis(ax):
    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.9,
        color=ZERO_LINE_COLOR,
        zorder=1
    )

    ax.set_xlim(-0.30, 2.55)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        width=0.8,
        length=3.2,
        top=False,
        right=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(SPINE_WIDTH)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.grid(False)


def use_special_tick_labels(row, side, prefix):
    """
    Special figures:
    1. crisis2015_sentiment_first-time asylum applicant_mentioned.png
    2. 2022_sentiment_first-time asylum applicant_mentioned.png
    """
    outcome = str(row["outcome"])
    treatment = str(row["treatment"])

    cond1 = (
        prefix == "crisis2015"
        and outcome == SPECIAL_OUTCOME
        and treatment in SPECIAL_TREATMENTS
        and side == SPECIAL_SIDE
    )

    cond2 = (
        prefix == "2022"
        and outcome == SPECIAL_OUTCOME
        and treatment in SPECIAL_TREATMENTS
        and side == SPECIAL_SIDE
    )

    return cond1 or cond2


def relabel_top_bottom_yticks(ax, bottom_label="-0.08", top_label="0.08"):
    ymin, ymax = ax.get_ylim()
    ticks = ax.get_yticks()
    visible_ticks = [t for t in ticks if ymin <= t <= ymax]

    if len(visible_ticks) < 2:
        return

    bottom_tick = visible_ticks[0]
    top_tick = visible_ticks[-1]

    labels = []
    for t in visible_ticks:
        if np.isclose(t, bottom_tick):
            labels.append(bottom_label)
        elif np.isclose(t, top_tick):
            labels.append(top_label)
        else:
            labels.append(dynamic_formatter(t, None))

    ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))


def draw_coef_point(
    ax,
    x_value,
    y_value,
    yerr_lower,
    yerr_upper,
    ci_low,
    ci_high,
    color,
    zorder=4
):
    """
    Draw CI and coefficient point separately.

    Significant = filled circle.
    Non-significant = hollow circle.
    """
    if pd.isna(y_value):
        return

    if pd.notna(yerr_lower) and pd.notna(yerr_upper):
        ax.errorbar(
            [x_value],
            [y_value],
            yerr=[[yerr_lower], [yerr_upper]],
            fmt="none",
            ecolor=color,
            elinewidth=ERROR_LINE_WIDTH,
            capsize=0,
            alpha=1.0,
            zorder=zorder
        )

    is_sig = ci_excludes_zero(ci_low, ci_high)

    ax.scatter(
        [x_value],
        [y_value],
        s=MARKER_SIZE,
        facecolors=color if is_sig else "white",
        edgecolors=color,
        linewidths=MARKER_EDGE_WIDTH,
        alpha=1.0,
        zorder=zorder + 1
    )


def outcome_to_filename_label(outcome):
    outcome = str(outcome)

    if outcome == "mean_sentiment_score":
        return "sentiment"
    elif outcome == "mean_hate_speech_score":
        return "hate speech"

    return sanitize_filename(outcome)


def outcome_axis_label(outcome, treatment):
    """Journal-style y-axis title with the outcome unit stated explicitly."""
    if str(outcome) == "mean_sentiment_score":
        return "Average change in sentiment\n(scale points)"
    if str(outcome) == "mean_hate_speech_score":
        return "Average change in hate speech\n(scale points)"
    return "Estimated change"


def treatment_to_filename_label(treatment):
    treatment = str(treatment)

    if treatment == "non-Ukrainian_first_time_applicant":
        return "non-Ukrainian first-time asylum applicant"
    if treatment == "first_time_applicants":
        return "first-time asylum applicant"
    elif treatment == "first_time_applicants_total":
        return "first-time asylum applicant (alternative total field)"
    elif treatment == "first_time_applicants_ukrainian":
        return "first-time asylum applicant (Ukrainians)"

    return sanitize_filename(treatment)


def make_figure_filename(prefix, outcome, treatment, side):
    """
    Required examples:
    2022_hate speech_first-time asylum applicant (Ukrainians)_origin.png
    crisis2015_sentiment_first-time asylum applicant_mentioned.png
    """
    crisis_label = prefix
    outcome_label = outcome_to_filename_label(outcome)
    treatment_label = treatment_to_filename_label(treatment)
    side_label = str(side)

    fname = f"{crisis_label}_{outcome_label}_{treatment_label}_{side_label}.png"
    return sanitize_filename(fname)


# ============================================================
# LOAD PERIOD PANEL
# ============================================================
def read_panel_from_period_file(combined_path: str, sheets, ycol: str, min_count: int, include_self: bool):
    data_dict = pd.read_excel(combined_path, sheet_name=sheets)
    all_rows = []

    for sh in sheets:
        df = data_dict[sh].copy()

        need_cols = ["origin_country", "mentioned_country", "post_count", ycol]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{sh}] missing columns: {missing}")

        df["origin_country"] = df["origin_country"].map(clean_country_name)
        df["mentioned_country"] = df["mentioned_country"].map(clean_country_name)
        df["post_count"] = pd.to_numeric(df["post_count"], errors="coerce")
        df[ycol] = pd.to_numeric(df[ycol], errors="coerce")

        df = df[df["post_count"] >= min_count].copy()
        df = df[~df[ycol].isna()].copy()

        df = df.rename(columns={
            "origin_country": "origin",
            "mentioned_country": "mentioned",
            ycol: "Y"
        })

        if not include_self:
            df = df[df["origin"] != df["mentioned"]].copy()

        df["period"] = sh
        df["pair"] = df["origin"] + "->" + df["mentioned"]
        df = df.drop_duplicates(subset=["pair", "period"])

        all_rows.append(df[["pair", "origin", "mentioned", "period", "Y", "post_count"]])

    panel = pd.concat(all_rows, ignore_index=True)
    return panel


# ============================================================
# YEARLY FILE HELPERS
# ============================================================
def _find_year_col(df):
    for c in ["Year", "year", "YEAR"]:
        if c in df.columns:
            return c
    return None


def _standardize_yearly_df(df: pd.DataFrame, year_value=None):
    df = df.copy()

    need_some_country_cols = ["origin_country", "mentioned_country"]
    missing_country = [c for c in need_some_country_cols if c not in df.columns]
    if missing_country:
        raise ValueError(f"Yearly data missing columns: {missing_country}")

    df["origin_country"] = df["origin_country"].map(clean_country_name)
    df["mentioned_country"] = df["mentioned_country"].map(clean_country_name)

    needed_numeric = [
        "positive_post_count", "neutral_post_count", "negative_post_count",
        "hate_speech_post_count", "normal_post_count", "offensive_post_count"
    ]
    missing_numeric = [c for c in needed_numeric if c not in df.columns]
    if missing_numeric:
        raise ValueError(f"Yearly data missing columns: {missing_numeric}")

    for c in needed_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if year_value is not None:
        df["year"] = int(year_value)
    else:
        yc = _find_year_col(df)
        if yc is None:
            raise ValueError("Yearly file must have per-year sheets or a Year/year column.")
        df["year"] = pd.to_numeric(df[yc], errors="coerce")

    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    return df


def load_yearly_long(yearly_path: str, target_years):
    xls = pd.ExcelFile(yearly_path)
    sheet_names = xls.sheet_names

    year_sheets_found = []
    for sh in sheet_names:
        s = str(sh).strip()
        if s.isdigit() and int(s) in target_years:
            year_sheets_found.append(int(s))

    if set(year_sheets_found) == set(target_years):
        parts = []
        for yy in target_years:
            tmp = pd.read_excel(yearly_path, sheet_name=str(yy))
            tmp = _standardize_yearly_df(tmp, year_value=yy)
            parts.append(tmp)
        out = pd.concat(parts, ignore_index=True)
        return out

    for sh in sheet_names:
        tmp = pd.read_excel(yearly_path, sheet_name=sh)
        yc = _find_year_col(tmp)
        needed = [
            "origin_country", "mentioned_country",
            "positive_post_count", "neutral_post_count", "negative_post_count",
            "hate_speech_post_count", "normal_post_count", "offensive_post_count"
        ]
        if yc is not None and all(c in tmp.columns for c in needed):
            tmp = _standardize_yearly_df(tmp, year_value=None)
            tmp = tmp[tmp["year"].isin(target_years)].copy()
            return tmp

    raise ValueError("Cannot parse yearly file correctly.")


# ============================================================
# BUILD PRE PANEL - 2022 DID VERSION
# ============================================================
def _map_2022_pre_period(y):
    y = int(y)
    if 2010 <= y <= 2014:
        return "2010-2014"
    elif 2015 <= y <= 2021:
        return "2015-2021"
    return np.nan


def build_pre_panel_2022_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
    target_years = list(range(2010, 2022))
    df = load_yearly_long(yearly_path, target_years).copy()

    df = df.rename(columns={"origin_country": "origin", "mentioned_country": "mentioned"})

    if not include_self:
        df = df[df["origin"] != df["mentioned"]].copy()

    df["pair"] = df["origin"] + "->" + df["mentioned"]
    df["period"] = df["year"].map(_map_2022_pre_period)
    df = df.dropna(subset=["period"]).copy()

    agg = (
        df.groupby(["pair", "origin", "mentioned", "period"], as_index=False)
        .agg({
            "positive_post_count": "sum",
            "neutral_post_count": "sum",
            "negative_post_count": "sum",
            "hate_speech_post_count": "sum",
            "normal_post_count": "sum",
            "offensive_post_count": "sum",
            "year": "nunique"
        })
        .rename(columns={"year": "n_years"})
    )

    sent_den = agg["positive_post_count"] + agg["neutral_post_count"] + agg["negative_post_count"]
    hate_den = agg["hate_speech_post_count"] + agg["normal_post_count"] + agg["offensive_post_count"]

    if ycol == "mean_sentiment_score":
        agg["raw_num"] = agg["positive_post_count"] - agg["negative_post_count"]
        agg["raw_den"] = sent_den
        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
        agg["post_count"] = agg["raw_den"]
    elif ycol == "mean_hate_speech_score":
        agg["raw_num"] = agg["hate_speech_post_count"] * 2 + agg["offensive_post_count"]
        agg["raw_den"] = hate_den
        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
        agg["post_count"] = agg["raw_den"]
    else:
        raise ValueError(f"Unsupported ycol: {ycol}")

    agg = agg[agg["post_count"] >= min_count].copy()
    agg = agg[~agg["Y"].isna()].copy()

    return agg[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den", "n_years"]]


# ============================================================
# BUILD PRE PANEL - 2015 CRISIS VERSION
# ============================================================
def _map_2015_pre_period(y):
    y = int(y)
    if y in [2010, 2011, 2012]:
        return "2010-2012"
    elif y in [2013, 2014]:
        return "2013-2014"
    return np.nan


def build_pre_panel_2015_from_yearly(yearly_path: str, ycol: str, min_count: int, include_self: bool):
    target_years = [2010, 2011, 2012, 2013, 2014]
    df = load_yearly_long(yearly_path, target_years).copy()

    df = df.rename(columns={"origin_country": "origin", "mentioned_country": "mentioned"})

    if not include_self:
        df = df[df["origin"] != df["mentioned"]].copy()

    df["pair"] = df["origin"] + "->" + df["mentioned"]
    df["period"] = df["year"].map(_map_2015_pre_period)
    df = df.dropna(subset=["period"]).copy()

    agg = (
        df.groupby(["pair", "origin", "mentioned", "period"], as_index=False)
        .agg({
            "positive_post_count": "sum",
            "neutral_post_count": "sum",
            "negative_post_count": "sum",
            "hate_speech_post_count": "sum",
            "normal_post_count": "sum",
            "offensive_post_count": "sum",
            "year": "nunique"
        })
        .rename(columns={"year": "n_years"})
    )

    sent_den = agg["positive_post_count"] + agg["neutral_post_count"] + agg["negative_post_count"]
    hate_den = agg["hate_speech_post_count"] + agg["normal_post_count"] + agg["offensive_post_count"]

    if ycol == "mean_sentiment_score":
        agg["raw_num"] = agg["positive_post_count"] - agg["negative_post_count"]
        agg["raw_den"] = sent_den
        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
        agg["post_count"] = agg["raw_den"]
    elif ycol == "mean_hate_speech_score":
        agg["raw_num"] = agg["hate_speech_post_count"] * 2 + agg["offensive_post_count"]
        agg["raw_den"] = hate_den
        agg["Y"] = np.where(agg["raw_den"] > 0, agg["raw_num"] / agg["raw_den"], np.nan)
        agg["post_count"] = agg["raw_den"]
    else:
        raise ValueError(f"Unsupported ycol: {ycol}")

    agg = agg[agg["post_count"] >= min_count].copy()
    agg = agg[~agg["Y"].isna()].copy()

    return agg[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den", "n_years"]]


# ============================================================
# TREATMENT HELPERS
# ============================================================
def make_treat_k(asylum_path: str, year: int, treat_col: str):
    a = pd.read_excel(asylum_path, sheet_name=str(year)).copy()

    need_cols = ["host_country", treat_col, "population"]
    missing = [c for c in need_cols if c not in a.columns]
    if missing:
        raise ValueError(f"[asylum {year}] missing columns: {missing}")

    a = a.rename(columns={"host_country": "country"})
    a["country"] = a["country"].map(clean_country_name)

    a[treat_col] = pd.to_numeric(a[treat_col], errors="coerce")
    a["population"] = pd.to_numeric(a["population"], errors="coerce")

    treat_k_col = f"treat_k_{year}"
    raw_treat_k_col = f"raw_treat_k_{year}"
    raw_rate = 1000.0 * a[treat_col] / a["population"]
    a[raw_treat_k_col] = raw_rate

    if (raw_rate.dropna() < 0).any():
        raise ValueError(
            f"[{treat_col}, {year}] negative treatment rates are incompatible "
            "with the log1p transformation."
        )

    if TREATMENT_TRANSFORM == "log1p":
        a[treat_k_col] = np.log1p(raw_rate)
    elif TREATMENT_TRANSFORM == "raw":
        a[treat_k_col] = raw_rate
    else:
        raise ValueError(
            f"Unknown TREATMENT_TRANSFORM: {TREATMENT_TRANSFORM}"
        )

    return a[["country", treat_k_col, raw_treat_k_col]]


def attach_treatment_two_sides(panel: pd.DataFrame, asylum_path: str, treat_col: str, treat_year: int):
    t = make_treat_k(asylum_path, treat_year, treat_col=treat_col)
    tk = f"treat_k_{treat_year}"
    raw_tk = f"raw_treat_k_{treat_year}"

    panel = panel.merge(
        t.rename(columns={
            "country": "origin",
            tk: "treat_origin",
            raw_tk: "raw_treat_origin",
        }),
        on="origin",
        how="left"
    )
    panel = panel.merge(
        t.rename(columns={
            "country": "mentioned",
            tk: "treat_mentioned",
            raw_tk: "raw_treat_mentioned",
        }),
        on="mentioned",
        how="left"
    )

    panel["treat_origin"] = pd.to_numeric(panel["treat_origin"], errors="coerce")
    panel["treat_mentioned"] = pd.to_numeric(panel["treat_mentioned"], errors="coerce")
    panel["raw_treat_origin"] = pd.to_numeric(panel["raw_treat_origin"], errors="coerce")
    panel["raw_treat_mentioned"] = pd.to_numeric(panel["raw_treat_mentioned"], errors="coerce")
    return panel


def attach_dyadic_treat_k_2015(panel: pd.DataFrame, asylum_path: str, treat_col: str):
    t15 = make_treat_k(asylum_path, 2015, treat_col=treat_col)

    panel = panel.merge(
        t15.rename(columns={
            "country": "origin",
            "treat_k_2015": "treat_origin_2015",
            "raw_treat_k_2015": "raw_treat_origin_2015",
        }),
        on="origin", how="left"
    )
    panel = panel.merge(
        t15.rename(columns={
            "country": "mentioned",
            "treat_k_2015": "treat_mentioned_2015",
            "raw_treat_k_2015": "raw_treat_mentioned_2015",
        }),
        on="mentioned", how="left"
    )

    panel["treat_origin_2015"] = pd.to_numeric(panel["treat_origin_2015"], errors="coerce")
    panel["treat_mentioned_2015"] = pd.to_numeric(panel["treat_mentioned_2015"], errors="coerce")
    panel["raw_treat_origin_2015"] = pd.to_numeric(panel["raw_treat_origin_2015"], errors="coerce")
    panel["raw_treat_mentioned_2015"] = pd.to_numeric(panel["raw_treat_mentioned_2015"], errors="coerce")
    return panel


# ============================================================
# VARIABLE BUILDERS
# ============================================================
def build_pretrend_vars_2022(panel: pd.DataFrame):
    panel = panel.copy()
    panel["period_id"] = panel["period"].map({"2010-2014": 0, "2015-2021": 1})
    panel["PrePeriod"] = (panel["period"] == "2010-2014").astype(int)
    panel["PRE_origin"] = panel["treat_origin"] * panel["PrePeriod"]
    panel["PRE_mentioned"] = panel["treat_mentioned"] * panel["PrePeriod"]
    return panel


def build_post_did_vars_2022(panel: pd.DataFrame):
    panel = panel.copy()
    panel["period_id"] = panel["period"].map({"2015-2021": 0, "2022-2024": 1})
    panel["PostPeriod"] = (panel["period"] == "2022-2024").astype(int)
    panel["POST_origin"] = panel["treat_origin"] * panel["PostPeriod"]
    panel["POST_mentioned"] = panel["treat_mentioned"] * panel["PostPeriod"]
    return panel


def build_pretrend_vars_2015(panel: pd.DataFrame):
    panel = panel.copy()
    panel["period_id"] = panel["period"].map({"2010-2012": 0, "2013-2014": 1})
    panel["PrePeriod"] = (panel["period"] == "2010-2012").astype(int)
    panel["PRE_origin"] = panel["treat_origin_2015"] * panel["PrePeriod"]
    panel["PRE_mentioned"] = panel["treat_mentioned_2015"] * panel["PrePeriod"]
    return panel


def build_post_did_vars_2015(panel: pd.DataFrame):
    panel = panel.copy()
    panel["period_id"] = panel["period"].map({"2013-2014": 0, "2015-2021": 1})
    panel["Post2015"] = (panel["period"] == "2015-2021").astype(int)
    panel["POST_origin"] = panel["treat_origin_2015"] * panel["Post2015"]
    panel["POST_mentioned"] = panel["treat_mentioned_2015"] * panel["Post2015"]
    return panel


# ============================================================
# REGRESSION
# ============================================================
def run_panel_ols(panel: pd.DataFrame, xcols, cluster_mode: str = "pair", use_weights: bool = False):
    df = panel.copy().set_index(["pair", "period_id"]).sort_index()

    y = df["Y"]
    X = df[xcols]

    model_kwargs = dict(
        dependent=y,
        exog=X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )

    if use_weights:
        model_kwargs["weights"] = pd.to_numeric(df["post_count"], errors="coerce").astype(float)

    model = PanelOLS(**model_kwargs)

    if cluster_mode == "pair":
        res = model.fit(cov_type="clustered", cluster_entity=True)
    elif cluster_mode == "origin_mentioned":
        clusters = pd.DataFrame({
            "origin": df["origin"].astype("category"),
            "mentioned": df["mentioned"].astype("category")
        }, index=df.index)
        res = model.fit(cov_type="clustered", clusters=clusters)
    else:
        raise ValueError("Unknown cluster_mode")

    return res, df


# ============================================================
# FINAL SUMMARY ROWS
# ============================================================
def make_final_summary_row_2022(ycol, treat_col, pre_coef_df, post_coef_df, pre_panel, post_panel):
    pre_origin = extract_term_or_nan(pre_coef_df, "PRE_origin")
    pre_mentioned = extract_term_or_nan(pre_coef_df, "PRE_mentioned")
    post_origin = extract_term_or_nan(post_coef_df, "POST_origin")
    post_mentioned = extract_term_or_nan(post_coef_df, "POST_mentioned")

    origin_parallel = "pass" if pd.notna(pre_origin["p"]) and pre_origin["p"] >= 0.05 else "fail"
    mentioned_parallel = "pass" if pd.notna(pre_mentioned["p"]) and pre_mentioned["p"] >= 0.05 else "fail"
    overall_parallel = "pass_both_sides" if origin_parallel == "pass" and mentioned_parallel == "pass" else "fail_at_least_one_side"

    dyad_effect_fields = {}
    dyad_effect_fields.update(dyad_effect_summary_fields(
        pre_panel, "origin", "raw_treat_origin", pre_origin["model_coef"],
        "origin_pre",
    ))
    dyad_effect_fields.update(dyad_effect_summary_fields(
        pre_panel, "mentioned", "raw_treat_mentioned", pre_mentioned["model_coef"],
        "mentioned_pre",
    ))
    dyad_effect_fields.update(dyad_effect_summary_fields(
        post_panel, "origin", "raw_treat_origin", post_origin["model_coef"],
        "origin_post",
    ))
    dyad_effect_fields.update(dyad_effect_summary_fields(
        post_panel, "mentioned", "raw_treat_mentioned", post_mentioned["model_coef"],
        "mentioned_post",
    ))

    return {
        "outcome": ycol,
        "treatment": treat_col,
        "model_treatment_scale": "log(1 + first-time asylum applicants per 1,000 population)",
        "reported_effect_scale": "sample-average discrete effect of +1 applicant per 1,000 residents",
        "estimator": "two-period pair and time fixed-effects DID",
        "covariance": "pair-clustered standard errors",

        "n_pairs_pre_total": pre_panel["pair"].nunique(),
        "n_obs_pre": len(pre_panel),
        "n_pairs_post_common": post_panel["pair"].nunique(),
        "n_obs_post": len(post_panel),

        "origin_parallel": origin_parallel,
        "mentioned_parallel": mentioned_parallel,
        "overall_parallel": overall_parallel,

        "origin_pre_period": "2010-2014",
        "origin_pre_event_time": -2,
        "origin_pre_coef_2010_2014_vs_2015_2021": pre_origin["coef"],
        "origin_pre_model_log1p_coef_2010_2014_vs_2015_2021": pre_origin["model_coef"],
        "origin_pre_average_discrete_factor": pre_origin["average_discrete_factor"],
        "origin_pre_se_2010_2014_vs_2015_2021": pre_origin["se"],
        "origin_pre_p_2010_2014_vs_2015_2021": pre_origin["p"],
        "origin_pre_ci_low_2010_2014_vs_2015_2021": pre_origin["ci_low"],
        "origin_pre_ci_high_2010_2014_vs_2015_2021": pre_origin["ci_high"],

        "origin_base_period": "2015-2021",
        "origin_base_event_time": -1,
        "origin_base_coef_2015_2021": 0.0,
        "origin_base_se_2015_2021": 0.0,
        "origin_base_p_2015_2021": np.nan,
        "origin_base_ci_low_2015_2021": 0.0,
        "origin_base_ci_high_2015_2021": 0.0,

        "origin_post_period": "2022-2024",
        "origin_post_event_time": 1,
        "origin_post_coef_2022_2024_vs_2015_2021": post_origin["coef"],
        "origin_post_model_log1p_coef_2022_2024_vs_2015_2021": post_origin["model_coef"],
        "origin_post_average_discrete_factor": post_origin["average_discrete_factor"],
        "origin_post_se_2022_2024_vs_2015_2021": post_origin["se"],
        "origin_post_p_2022_2024_vs_2015_2021": post_origin["p"],
        "origin_post_ci_low_2022_2024_vs_2015_2021": post_origin["ci_low"],
        "origin_post_ci_high_2022_2024_vs_2015_2021": post_origin["ci_high"],

        "mentioned_pre_period": "2010-2014",
        "mentioned_pre_event_time": -2,
        "mentioned_pre_coef_2010_2014_vs_2015_2021": pre_mentioned["coef"],
        "mentioned_pre_model_log1p_coef_2010_2014_vs_2015_2021": pre_mentioned["model_coef"],
        "mentioned_pre_average_discrete_factor": pre_mentioned["average_discrete_factor"],
        "mentioned_pre_se_2010_2014_vs_2015_2021": pre_mentioned["se"],
        "mentioned_pre_p_2010_2014_vs_2015_2021": pre_mentioned["p"],
        "mentioned_pre_ci_low_2010_2014_vs_2015_2021": pre_mentioned["ci_low"],
        "mentioned_pre_ci_high_2010_2014_vs_2015_2021": pre_mentioned["ci_high"],

        "mentioned_base_period": "2015-2021",
        "mentioned_base_event_time": -1,
        "mentioned_base_coef_2015_2021": 0.0,
        "mentioned_base_se_2015_2021": 0.0,
        "mentioned_base_p_2015_2021": np.nan,
        "mentioned_base_ci_low_2015_2021": 0.0,
        "mentioned_base_ci_high_2015_2021": 0.0,

        "mentioned_post_period": "2022-2024",
        "mentioned_post_event_time": 1,
        "mentioned_post_coef_2022_2024_vs_2015_2021": post_mentioned["coef"],
        "mentioned_post_model_log1p_coef_2022_2024_vs_2015_2021": post_mentioned["model_coef"],
        "mentioned_post_average_discrete_factor": post_mentioned["average_discrete_factor"],
        "mentioned_post_se_2022_2024_vs_2015_2021": post_mentioned["se"],
        "mentioned_post_p_2022_2024_vs_2015_2021": post_mentioned["p"],
        "mentioned_post_ci_low_2022_2024_vs_2015_2021": post_mentioned["ci_low"],
        "mentioned_post_ci_high_2022_2024_vs_2015_2021": post_mentioned["ci_high"],
        **dyad_effect_fields,
    }


def make_final_summary_row_2015(ycol, treat_col, pre_coef_df, post_coef_df, pre_panel, post_panel):
    pre_origin = extract_term_or_nan(pre_coef_df, "PRE_origin")
    pre_mentioned = extract_term_or_nan(pre_coef_df, "PRE_mentioned")
    post_origin = extract_term_or_nan(post_coef_df, "POST_origin")
    post_mentioned = extract_term_or_nan(post_coef_df, "POST_mentioned")

    origin_parallel = "pass" if pd.notna(pre_origin["p"]) and pre_origin["p"] >= 0.05 else "fail"
    mentioned_parallel = "pass" if pd.notna(pre_mentioned["p"]) and pre_mentioned["p"] >= 0.05 else "fail"
    overall_parallel = "pass_both_sides" if origin_parallel == "pass" and mentioned_parallel == "pass" else "fail_at_least_one_side"

    dyad_effect_fields = {}
    dyad_effect_fields.update(dyad_effect_summary_fields(
        pre_panel, "origin", "raw_treat_origin_2015", pre_origin["model_coef"],
        "origin_pre",
    ))
    dyad_effect_fields.update(dyad_effect_summary_fields(
        pre_panel, "mentioned", "raw_treat_mentioned_2015", pre_mentioned["model_coef"],
        "mentioned_pre",
    ))
    dyad_effect_fields.update(dyad_effect_summary_fields(
        post_panel, "origin", "raw_treat_origin_2015", post_origin["model_coef"],
        "origin_post",
    ))
    dyad_effect_fields.update(dyad_effect_summary_fields(
        post_panel, "mentioned", "raw_treat_mentioned_2015", post_mentioned["model_coef"],
        "mentioned_post",
    ))

    return {
        "outcome": ycol,
        "treatment": treat_col,
        "model_treatment_scale": "log(1 + first-time asylum applicants per 1,000 population)",
        "reported_effect_scale": "sample-average discrete effect of +1 applicant per 1,000 residents",
        "estimator": "two-period pair and time fixed-effects DID",
        "covariance": "pair-clustered standard errors",

        "n_pairs_pre_total": pre_panel["pair"].nunique(),
        "n_obs_pre": len(pre_panel),
        "n_pairs_post_common": post_panel["pair"].nunique(),
        "n_obs_post": len(post_panel),

        "origin_parallel": origin_parallel,
        "mentioned_parallel": mentioned_parallel,
        "overall_parallel": overall_parallel,

        "origin_pre_period": "2010-2012",
        "origin_pre_event_time": -2,
        "origin_pre_coef_2010_2012_vs_2013_2014": pre_origin["coef"],
        "origin_pre_model_log1p_coef_2010_2012_vs_2013_2014": pre_origin["model_coef"],
        "origin_pre_average_discrete_factor": pre_origin["average_discrete_factor"],
        "origin_pre_se_2010_2012_vs_2013_2014": pre_origin["se"],
        "origin_pre_p_2010_2012_vs_2013_2014": pre_origin["p"],
        "origin_pre_ci_low_2010_2012_vs_2013_2014": pre_origin["ci_low"],
        "origin_pre_ci_high_2010_2012_vs_2013_2014": pre_origin["ci_high"],

        "origin_base_period": "2013-2014",
        "origin_base_event_time": -1,
        "origin_base_coef_2013_2014": 0.0,
        "origin_base_se_2013_2014": 0.0,
        "origin_base_p_2013_2014": np.nan,
        "origin_base_ci_low_2013_2014": 0.0,
        "origin_base_ci_high_2013_2014": 0.0,

        "origin_post_period": "2015-2021",
        "origin_post_event_time": 1,
        "origin_post_coef_2015_2021_vs_2013_2014": post_origin["coef"],
        "origin_post_model_log1p_coef_2015_2021_vs_2013_2014": post_origin["model_coef"],
        "origin_post_average_discrete_factor": post_origin["average_discrete_factor"],
        "origin_post_se_2015_2021_vs_2013_2014": post_origin["se"],
        "origin_post_p_2015_2021_vs_2013_2014": post_origin["p"],
        "origin_post_ci_low_2015_2021_vs_2013_2014": post_origin["ci_low"],
        "origin_post_ci_high_2015_2021_vs_2013_2014": post_origin["ci_high"],

        "mentioned_pre_period": "2010-2012",
        "mentioned_pre_event_time": -2,
        "mentioned_pre_coef_2010_2012_vs_2013_2014": pre_mentioned["coef"],
        "mentioned_pre_model_log1p_coef_2010_2012_vs_2013_2014": pre_mentioned["model_coef"],
        "mentioned_pre_average_discrete_factor": pre_mentioned["average_discrete_factor"],
        "mentioned_pre_se_2010_2012_vs_2013_2014": pre_mentioned["se"],
        "mentioned_pre_p_2010_2012_vs_2013_2014": pre_mentioned["p"],
        "mentioned_pre_ci_low_2010_2012_vs_2013_2014": pre_mentioned["ci_low"],
        "mentioned_pre_ci_high_2010_2012_vs_2013_2014": pre_mentioned["ci_high"],

        "mentioned_base_period": "2013-2014",
        "mentioned_base_event_time": -1,
        "mentioned_base_coef_2013_2014": 0.0,
        "mentioned_base_se_2013_2014": 0.0,
        "mentioned_base_p_2013_2014": np.nan,
        "mentioned_base_ci_low_2013_2014": 0.0,
        "mentioned_base_ci_high_2013_2014": 0.0,

        "mentioned_post_period": "2015-2021",
        "mentioned_post_event_time": 1,
        "mentioned_post_coef_2015_2021_vs_2013_2014": post_mentioned["coef"],
        "mentioned_post_model_log1p_coef_2015_2021_vs_2013_2014": post_mentioned["model_coef"],
        "mentioned_post_average_discrete_factor": post_mentioned["average_discrete_factor"],
        "mentioned_post_se_2015_2021_vs_2013_2014": post_mentioned["se"],
        "mentioned_post_p_2015_2021_vs_2013_2014": post_mentioned["p"],
        "mentioned_post_ci_low_2015_2021_vs_2013_2014": post_mentioned["ci_low"],
        "mentioned_post_ci_high_2015_2021_vs_2013_2014": post_mentioned["ci_high"],
        **dyad_effect_fields,
    }


# ============================================================
# RUN ONE SPEC - 2022 DID VERSION
# ============================================================
def run_one_spec_2022(combined_path, yearly_path, asylum_path, ycol, treat_col,
                      min_count=30, include_self=False, use_weights=False):

    # A. PRE-TREND: 2010-2014 vs 2015-2021
    pre_panel = build_pre_panel_2022_from_yearly(
        yearly_path=yearly_path,
        ycol=ycol,
        min_count=min_count,
        include_self=include_self
    )
    pre_panel = attach_treatment_two_sides(
        panel=pre_panel,
        asylum_path=asylum_path,
        treat_col=treat_col,
        treat_year=2022
    )
    pre_panel = pre_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
    pre_panel, _ = keep_common_pairs(pre_panel, ["2010-2014", "2015-2021"])
    pre_panel = build_pretrend_vars_2022(pre_panel)

    pre_res, _ = run_panel_ols(
        pre_panel,
        ["PRE_origin", "PRE_mentioned"],
        "pair",
        use_weights
    )
    pre_coef = coef_table_from_res(pre_res)
    pre_coef = convert_coef_table_to_average_discrete_effect(
        pre_coef,
        pre_panel,
        {
            "PRE_origin": "raw_treat_origin",
            "PRE_mentioned": "raw_treat_mentioned",
        },
    )

    # B. POST-DID: 2015-2021 vs 2022-2024
    post_panel = read_panel_from_period_file(
        combined_path=combined_path,
        sheets=["2015-2021", "2022-2024"],
        ycol=ycol,
        min_count=min_count,
        include_self=include_self
    )
    post_panel, _ = keep_common_pairs(post_panel, ["2015-2021", "2022-2024"])
    post_panel = attach_treatment_two_sides(
        panel=post_panel,
        asylum_path=asylum_path,
        treat_col=treat_col,
        treat_year=2022
    )
    post_panel = post_panel.dropna(subset=["treat_origin", "treat_mentioned"]).copy()
    post_panel = build_post_did_vars_2022(post_panel)

    post_res, _ = run_panel_ols(
        post_panel,
        ["POST_origin", "POST_mentioned"],
        "pair",
        use_weights
    )
    post_coef = coef_table_from_res(post_res)
    post_coef = convert_coef_table_to_average_discrete_effect(
        post_coef,
        post_panel,
        {
            "POST_origin": "raw_treat_origin",
            "POST_mentioned": "raw_treat_mentioned",
        },
    )

    return make_final_summary_row_2022(
        ycol=ycol,
        treat_col=treat_col,
        pre_coef_df=pre_coef,
        post_coef_df=post_coef,
        pre_panel=pre_panel,
        post_panel=post_panel
    )


# ============================================================
# RUN ONE SPEC - 2015 CRISIS VERSION
# ============================================================
def run_one_spec_2015(combined_path, yearly_path, asylum_path, ycol, treat_col,
                      min_count=20, include_self=False, use_weights=False):

    # A. PRE-TREND: 2010-2012 vs 2013-2014
    pre_panel = build_pre_panel_2015_from_yearly(
        yearly_path=yearly_path,
        ycol=ycol,
        min_count=min_count,
        include_self=include_self
    )
    pre_panel = attach_dyadic_treat_k_2015(pre_panel, asylum_path, treat_col=treat_col)
    pre_panel = pre_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
    pre_panel = build_pretrend_vars_2015(pre_panel)

    pre_res, _ = run_panel_ols(
        pre_panel,
        ["PRE_origin", "PRE_mentioned"],
        "pair",
        use_weights
    )
    pre_coef = coef_table_from_res(pre_res)
    pre_coef = convert_coef_table_to_average_discrete_effect(
        pre_coef,
        pre_panel,
        {
            "PRE_origin": "raw_treat_origin_2015",
            "PRE_mentioned": "raw_treat_mentioned_2015",
        },
    )

    # B. DID POST: 2013-2014 vs 2015-2021
    baseline_panel = build_pre_panel_2015_from_yearly(
        yearly_path=yearly_path,
        ycol=ycol,
        min_count=min_count,
        include_self=include_self
    )
    baseline_panel = baseline_panel[baseline_panel["period"] == "2013-2014"].copy()

    post_period_panel = read_panel_from_period_file(
        combined_path=combined_path,
        sheets=["2015-2021"],
        ycol=ycol,
        min_count=min_count,
        include_self=include_self
    )

    post_panel = pd.concat([
        baseline_panel[["pair", "origin", "mentioned", "period", "Y", "post_count", "raw_num", "raw_den"]],
        post_period_panel[["pair", "origin", "mentioned", "period", "Y", "post_count"]]
    ], ignore_index=True)

    post_panel, _ = keep_common_pairs(post_panel, ["2013-2014", "2015-2021"])
    post_panel = attach_dyadic_treat_k_2015(post_panel, asylum_path, treat_col=treat_col)
    post_panel = post_panel.dropna(subset=["treat_origin_2015", "treat_mentioned_2015"]).copy()
    post_panel = build_post_did_vars_2015(post_panel)

    post_res, _ = run_panel_ols(
        post_panel,
        ["POST_origin", "POST_mentioned"],
        "pair",
        use_weights
    )
    post_coef = coef_table_from_res(post_res)
    post_coef = convert_coef_table_to_average_discrete_effect(
        post_coef,
        post_panel,
        {
            "POST_origin": "raw_treat_origin_2015",
            "POST_mentioned": "raw_treat_mentioned_2015",
        },
    )

    return make_final_summary_row_2015(
        ycol=ycol,
        treat_col=treat_col,
        pre_coef_df=pre_coef,
        post_coef_df=post_coef,
        pre_panel=pre_panel,
        post_panel=post_panel
    )


# ============================================================
# PERIOD LABELS FOR FIGURES
# ============================================================
def format_period_labels_type1(periods):
    mapping = {
        "2010-2014": "2010-2014\n(pre 1)",
        "2015-2021": "2015-2021\n(baseline)",
        "2022-2024": "2022-2024\n(post 1)",
    }
    return [mapping.get(p, str(p)) for p in periods]


def format_period_labels_type2(periods):
    mapping = {
        "2010-2012": "2010-2012\n(pre 1)",
        "2013-2014": "2013-2014\n(baseline)",
        "2015-2021": "2015-2021\n(post 1)",
    }
    return [mapping.get(p, str(p)) for p in periods]


# ============================================================
# GET SERIES - DATASET 1: 2022 DID
# ============================================================
def get_series_type1(row, side):
    x = unified_x()

    if side == "origin":
        periods = [
            row["origin_pre_period"],
            row["origin_base_period"],
            row["origin_post_period"]
        ]

        y = [
            row["origin_pre_coef_2010_2014_vs_2015_2021"],
            row["origin_base_coef_2015_2021"],
            row["origin_post_coef_2022_2024_vs_2015_2021"]
        ]

        ci_low = [
            row["origin_pre_ci_low_2010_2014_vs_2015_2021"],
            row["origin_base_ci_low_2015_2021"],
            row["origin_post_ci_low_2022_2024_vs_2015_2021"]
        ]

        ci_high = [
            row["origin_pre_ci_high_2010_2014_vs_2015_2021"],
            row["origin_base_ci_high_2015_2021"],
            row["origin_post_ci_high_2022_2024_vs_2015_2021"]
        ]

    else:
        periods = [
            row["mentioned_pre_period"],
            row["mentioned_base_period"],
            row["mentioned_post_period"]
        ]

        y = [
            row["mentioned_pre_coef_2010_2014_vs_2015_2021"],
            row["mentioned_base_coef_2015_2021"],
            row["mentioned_post_coef_2022_2024_vs_2015_2021"]
        ]

        ci_low = [
            row["mentioned_pre_ci_low_2010_2014_vs_2015_2021"],
            row["mentioned_base_ci_low_2015_2021"],
            row["mentioned_post_ci_low_2022_2024_vs_2015_2021"]
        ]

        ci_high = [
            row["mentioned_pre_ci_high_2010_2014_vs_2015_2021"],
            row["mentioned_base_ci_high_2015_2021"],
            row["mentioned_post_ci_high_2022_2024_vs_2015_2021"]
        ]

    yerr_lower = [
        y[i] - ci_low[i] if pd.notna(y[i]) and pd.notna(ci_low[i]) else np.nan
        for i in range(3)
    ]

    yerr_upper = [
        ci_high[i] - y[i] if pd.notna(y[i]) and pd.notna(ci_high[i]) else np.nan
        for i in range(3)
    ]

    return x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high


# ============================================================
# GET SERIES - DATASET 2: 2015 CRISIS
# ============================================================
def get_series_type2(row, side):
    x = unified_x()

    if side == "origin":
        periods = [
            row["origin_pre_period"],
            row["origin_base_period"],
            row["origin_post_period"]
        ]

        y = [
            row["origin_pre_coef_2010_2012_vs_2013_2014"],
            row["origin_base_coef_2013_2014"],
            row["origin_post_coef_2015_2021_vs_2013_2014"]
        ]

        ci_low = [
            row["origin_pre_ci_low_2010_2012_vs_2013_2014"],
            row["origin_base_ci_low_2013_2014"],
            row["origin_post_ci_low_2015_2021_vs_2013_2014"]
        ]

        ci_high = [
            row["origin_pre_ci_high_2010_2012_vs_2013_2014"],
            row["origin_base_ci_high_2013_2014"],
            row["origin_post_ci_high_2015_2021_vs_2013_2014"]
        ]

    else:
        periods = [
            row["mentioned_pre_period"],
            row["mentioned_base_period"],
            row["mentioned_post_period"]
        ]

        y = [
            row["mentioned_pre_coef_2010_2012_vs_2013_2014"],
            row["mentioned_base_coef_2013_2014"],
            row["mentioned_post_coef_2015_2021_vs_2013_2014"]
        ]

        ci_low = [
            row["mentioned_pre_ci_low_2010_2012_vs_2013_2014"],
            row["mentioned_base_ci_low_2013_2014"],
            row["mentioned_post_ci_low_2015_2021_vs_2013_2014"]
        ]

        ci_high = [
            row["mentioned_pre_ci_high_2010_2012_vs_2013_2014"],
            row["mentioned_base_ci_high_2013_2014"],
            row["mentioned_post_ci_high_2015_2021_vs_2013_2014"]
        ]

    yerr_lower = [
        y[i] - ci_low[i] if pd.notna(y[i]) and pd.notna(ci_low[i]) else np.nan
        for i in range(3)
    ]

    yerr_upper = [
        ci_high[i] - y[i] if pd.notna(y[i]) and pd.notna(ci_high[i]) else np.nan
        for i in range(3)
    ]

    return x, y, yerr_lower, yerr_upper, periods, ci_low, ci_high


# ============================================================
# DYAD-DISTRIBUTION FIGURE HELPERS
# ============================================================
def dyad_effect_frame_from_row(row, side, stage):
    field = f"{side}_{stage}_dyad_effects_json"
    value = row.get(field, "")
    if pd.isna(value) or str(value).strip() == "":
        return pd.DataFrame(
            columns=[
                "pair", "origin", "mentioned", "treatment_side",
                "treatment_country", "raw_treatment_per_1000",
                "model_scale_contrast", "dyad_discrete_effect",
            ]
        )
    records = json.loads(str(value))
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["dyad_discrete_effect"] = pd.to_numeric(
            frame["dyad_discrete_effect"], errors="coerce"
        )
        frame = frame.dropna(subset=["dyad_discrete_effect"]).copy()
    return frame


def dyad_effect_plot_data(row, side, dataset_type):
    if side == "origin":
        periods = [
            row["origin_pre_period"],
            row["origin_base_period"],
            row["origin_post_period"],
        ]
    else:
        periods = [
            row["mentioned_pre_period"],
            row["mentioned_base_period"],
            row["mentioned_post_period"],
        ]
    labels = (
        format_period_labels_type1(periods)
        if dataset_type == "2022"
        else format_period_labels_type2(periods)
    )
    return {
        "x": unified_x(),
        "periods": periods,
        "labels": labels,
        "pre": dyad_effect_frame_from_row(row, side, "pre"),
        "post": dyad_effect_frame_from_row(row, side, "post"),
    }


def post_did_p_value(row, side, dataset_type):
    if dataset_type == "2022":
        return row[f"{side}_post_p_2022_2024_vs_2015_2021"]
    return row[f"{side}_post_p_2015_2021_vs_2013_2014"]


def format_p_value(value):
    if pd.isna(value):
        return "p = n.a."
    value = float(value)
    if value < 0.001:
        return "p < .001"
    return f"p = {value:.3f}".replace("0.", ".")


def draw_dyad_distribution(ax, x_value, frame, color):
    """Draw a modified box plot whose center line is the dyad mean.

    The box spans Q25-Q75 and the whiskers span the observed minimum and
    maximum.  Deliberately omit dyad dots and a separate mean marker.
    """
    values = frame["dyad_discrete_effect"].to_numpy(dtype=float)
    if len(values) == 0:
        return None

    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    mean = float(np.mean(values))
    stats = [{
        "label": "",
        "whislo": minimum,
        "q1": float(q25),
        # bxp calls this field "med"; setting it to the mean makes the
        # center line represent the dyad mean requested for the
        # figure.  The actual median is retained in the returned summary.
        "med": mean,
        "q3": float(q75),
        "whishi": maximum,
        "fliers": [],
    }]
    ax.bxp(
        stats,
        positions=[x_value],
        widths=BOX_WIDTH,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={
            "facecolor": "none",
            "edgecolor": color,
            "linewidth": BOX_LINE_WIDTH,
        },
        # Hide bxp's full-width centre line and draw a shorter mean line below,
        # so it does not cover the two vertical sides of the box.
        medianprops={"color": "none", "linewidth": 0},
        whiskerprops={"color": color, "linewidth": BOX_LINE_WIDTH},
        capprops={"color": color, "linewidth": BOX_LINE_WIDTH},
        zorder=3,
    )
    # Meet the inner sides of the box exactly.  Butt caps prevent the mean
    # line from extending over the vertical box borders despite its width.
    mean_half_width = BOX_WIDTH * 0.50
    ax.plot(
        [x_value - mean_half_width, x_value + mean_half_width],
        [mean, mean],
        color="#222222",
        linewidth=BOX_LINE_WIDTH,
        solid_capstyle="butt",
        zorder=4,
    )
    return {
        "n": int(len(values)),
        "min": minimum,
        "q25": float(q25),
        "median": float(median),
        "mean": mean,
        "q75": float(q75),
        "max": maximum,
    }


def all_dyad_effect_values(row, side):
    values = []
    for stage in ["pre", "post"]:
        frame = dyad_effect_frame_from_row(row, side, stage)
        values.extend(frame["dyad_discrete_effect"].tolist())
    return [float(value) for value in values if pd.notna(value)]


# ============================================================
# SHARED Y-AXIS FOR THE TWO SPECIAL FIGURES
# ============================================================
def build_shared_special_ylim(df1, df2):
    values = []
    for data, dataset_type in [(df1, "2022"), (df2, "2015")]:
        subset = data[
            (data["outcome"].astype(str) == SPECIAL_OUTCOME)
            & (data["treatment"].astype(str).isin(SPECIAL_TREATMENTS))
        ]
        if len(subset) > 0:
            row = subset.iloc[0]
            if dataset_type == "2022":
                _, y, _, _, _, ci_low, ci_high = get_series_type1(row, SPECIAL_SIDE)
            else:
                _, y, _, _, _, ci_low, ci_high = get_series_type2(row, SPECIAL_SIDE)
            values.extend([v for v in y + ci_low + ci_high if pd.notna(v)])
    if not values:
        return None
    return nice_ylim_raw(values, values)


# ============================================================
# DRAW SINGLE SEPARATE-SCALE COEFFICIENT / POST-BOX FIGURE
# ============================================================
def draw_single_coef_plot(row, side, dataset_type, prefix, output_dir, special_shared_ylim):
    outcome_value = row["outcome"]
    color = get_line_color(outcome_value)
    plot_data = dyad_effect_plot_data(row, side, dataset_type)
    x = plot_data["x"]

    if dataset_type == "2022":
        _, y, yerr_lower, yerr_upper, _, ci_low, ci_high = get_series_type1(row, side)
    else:
        _, y, yerr_lower, yerr_upper, _, ci_low, ci_high = get_series_type2(row, side)

    # Keep coefficient estimates and their CIs together on one scale.  Give the
    # much narrower dyad distribution a distinct, visibly separated scale.
    fig, (ax_coef, ax_box) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={
            "width_ratios": [ORIGINAL_MAIN_WIDTH, BOX_PANEL_WIDTH],
            "wspace": 0.085,
        },
    )

    # Left panel: original pre-period coefficient.
    draw_coef_point(
        ax=ax_coef,
        x_value=x[0],
        y_value=y[0],
        yerr_lower=yerr_lower[0],
        yerr_upper=yerr_upper[0],
        ci_low=ci_low[0],
        ci_high=ci_high[0],
        color=color,
    )

    # The omitted baseline category remains labelled on the x-axis and fixed
    # at zero, but is not drawn as a separate marker.

    # Left panel: original post-DID coefficient and 95% CI.
    draw_coef_point(
        ax=ax_coef,
        x_value=x[2],
        y_value=y[2],
        yerr_lower=yerr_lower[2],
        yerr_upper=yerr_upper[2],
        ci_low=ci_low[2],
        ci_high=ci_high[2],
        color=color,
    )

    # Right panel: post-period dyad distribution on its own expanded scale.
    # The center line is the dyad mean.  There are no dyad dots and
    # no separate mean marker.
    post_stats = draw_dyad_distribution(
        ax_box, 0.0, plot_data["post"], color
    )
    if post_stats is not None and not np.isclose(
        post_stats["mean"], float(y[2]), rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "Post dyad mean does not match the reported sample-average "
            f"DID effect: dyad mean={post_stats['mean']}, reported={y[2]}"
        )

    # Coefficient-panel scale: determined only by regression estimates/CIs.
    coef_values = [v for v in y + ci_low + ci_high if pd.notna(v)]
    is_special = use_special_tick_labels(row, side, prefix)
    if is_special and special_shared_ylim is not None:
        coef_ymin, coef_ymax = special_shared_ylim
    else:
        coef_ymin, coef_ymax = nice_ylim_raw(coef_values, coef_values)
    ax_coef.set_ylim(coef_ymin, coef_ymax)
    style_axis(ax_coef)

    if is_special and RELABEL_SPECIAL_YTICKS:
        relabel_top_bottom_yticks(
            ax_coef,
            bottom_label=SPECIAL_BOTTOM_LABEL,
            top_label=SPECIAL_TOP_LABEL,
        )

    ax_coef.set_xticks(x)
    ax_coef.set_xticklabels(plot_data["labels"])
    ax_coef.set_xlabel("")
    if SHOW_YLABEL:
        ax_coef.set_ylabel(
            outcome_axis_label(outcome_value, row["treatment"]),
            fontsize=LABEL_SIZE,
            labelpad=9,
        )

    # Box-panel scale: use only the post dyad distribution, without forcing
    # symmetry around zero.  This prevents the Q25-Q75 box from being flattened
    # by the much wider regression confidence intervals.
    box_values = plot_data["post"]["dyad_discrete_effect"].dropna().tolist()
    box_ymin, box_ymax = nice_ylim_distribution(box_values)
    ax_box.set_ylim(box_ymin, box_ymax)
    ax_box.set_xlim(-0.46, 0.46)
    ax_box.set_xticks([0.0])
    post_period_label = str(plot_data["periods"][2]).replace("-", "–")
    ax_box.set_xticklabels([post_period_label])
    ax_box.set_xlabel("")
    ax_box.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_box.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
    ax_box.yaxis.tick_right()
    ax_box.yaxis.set_label_position("right")
    ax_box.tick_params(
        axis="both",
        labelsize=max(TICK_SIZE - 1, 8),
        width=0.8,
        length=3.0,
        top=False,
        left=False,
        right=True,
    )
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["left"].set_visible(False)
    ax_box.spines["right"].set_linewidth(SPINE_WIDTH)
    ax_box.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax_box.spines["right"].set_color("#333333")
    ax_box.spines["bottom"].set_color("#333333")
    ax_box.set_facecolor("white")
    ax_box.set_title("")
    ax_box.set_ylabel(
        "Dyad estimates\n(scale points)",
        fontsize=max(LABEL_SIZE - 1, 9),
        labelpad=8,
    )

    if box_ymin <= 0 <= box_ymax:
        ax_box.axhline(
            0,
            linestyle="--",
            linewidth=0.9,
            color=ZERO_LINE_COLOR,
            zorder=1,
        )

    fname = make_figure_filename(
        prefix=prefix,
        outcome=row["outcome"],
        treatment=row["treatment"],
        side=side,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, fname),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def draw_all_refined_figures(results_2022_csv, results_2015_csv, figure_dir):
    safe_mkdir(figure_dir)

    df1 = pd.read_csv(results_2022_csv)
    df2 = pd.read_csv(results_2015_csv)

    special_shared_ylim = build_shared_special_ylim(df1, df2)

    # 2022 crisis figures
    for _, row in df1.iterrows():
        draw_single_coef_plot(row, "origin", "2022", "2022", figure_dir, special_shared_ylim)
        draw_single_coef_plot(row, "mentioned", "2022", "2022", figure_dir, special_shared_ylim)

    # 2015 crisis figures
    for _, row in df2.iterrows():
        draw_single_coef_plot(row, "origin", "crisis2015", "crisis2015", figure_dir, special_shared_ylim)
        draw_single_coef_plot(row, "mentioned", "crisis2015", "crisis2015", figure_dir, special_shared_ylim)


def export_dyad_specific_effects(results_csv_paths, output_path):
    """Export the exact dyad values summarized by every box-and-whisker figure."""
    rows = []
    for dataset_type, csv_path in results_csv_paths:
        results = pd.read_csv(csv_path)
        for _, result in results.iterrows():
            for side in ["origin", "mentioned"]:
                for stage in ["pre", "post"]:
                    frame = dyad_effect_frame_from_row(result, side, stage)
                    if frame.empty:
                        continue
                    period = result[f"{side}_{stage}_period"]
                    frame = frame.copy()
                    frame.insert(0, "dataset_type", dataset_type)
                    frame.insert(1, "outcome", result["outcome"])
                    frame.insert(2, "treatment", result["treatment"])
                    frame.insert(3, "side", side)
                    frame.insert(4, "stage", stage)
                    frame.insert(5, "period", period)
                    rows.append(frame)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


# ============================================================
# MAIN RUNNERS
# ============================================================
def run_2022crisis_version(master_out_dir):
    master_summary = []

    for outcome in DID2022_OUTCOME_SPECS:
        ycol = outcome["ycol"]

        for treat in DID2022_TREAT_SPECS:
            treat_col = treat["treat_col"]
            print(f"[2022 CRISIS] Running: Y={ycol}, Treat={treat_col}")

            summary_row = run_one_spec_2022(
                combined_path=combined_path,
                yearly_path=yearly_path,
                asylum_path=asylum_path,
                ycol=ycol,
                treat_col=treat_col,
                min_count=DID2022_MIN_COUNT,
                include_self=INCLUDE_SELF,
                use_weights=USE_WEIGHTS
            )
            master_summary.append(summary_row)

    out_csv = os.path.join(master_out_dir, "results_2022crisis.csv")
    pd.DataFrame(master_summary).to_csv(out_csv, index=False, encoding="utf-8-sig")

    return out_csv


def run_2015crisis_version(master_out_dir):
    master_summary = []

    for outcome in CRISIS2015_OUTCOME_SPECS:
        ycol = outcome["ycol"]

        for treat in CRISIS2015_TREAT_SPECS:
            treat_col = treat["treat_col"]
            print(f"[2015 CRISIS] Running: Y={ycol}, Treat={treat_col}")

            summary_row = run_one_spec_2015(
                combined_path=combined_path,
                yearly_path=yearly_path,
                asylum_path=asylum_path,
                ycol=ycol,
                treat_col=treat_col,
                min_count=CRISIS2015_MIN_COUNT,
                include_self=INCLUDE_SELF,
                use_weights=USE_WEIGHTS
            )
            master_summary.append(summary_row)

    out_csv = os.path.join(master_out_dir, "results_2015crisis.csv")
    pd.DataFrame(master_summary).to_csv(out_csv, index=False, encoding="utf-8-sig")

    return out_csv


# ============================================================
# MAIN
# ============================================================
def main():
    tag = now_tag()

    master_out_dir = os.path.join(
        OUT_ROOT,
        f"did_results_and_figures_{tag}"
    )
    safe_mkdir(master_out_dir)

    figure_dir = os.path.join(master_out_dir, "figures")
    safe_mkdir(figure_dir)

    # 1. Run DID and output CSV result tables only
    results_2022_csv = run_2022crisis_version(master_out_dir)
    results_2015_csv = run_2015crisis_version(master_out_dir)

    dyad_effects_csv = export_dyad_specific_effects(
        results_csv_paths=[
            ("2022", results_2022_csv),
            ("2015", results_2015_csv),
        ],
        output_path=os.path.join(
            master_out_dir, "dyad_specific_discrete_effects.csv"
        ),
    )

    # 2. Draw refined figures
    draw_all_refined_figures(
        results_2022_csv=results_2022_csv,
        results_2015_csv=results_2015_csv,
        figure_dir=figure_dir
    )

    print("Finished.")
    print("Master folder:", master_out_dir)
    print("2022 crisis CSV:", results_2022_csv)
    print("2015 crisis CSV:", results_2015_csv)
    print("Dyad-specific effects CSV:", dyad_effects_csv)
    print("Figures folder:", figure_dir)


if __name__ == "__main__":
    main()
