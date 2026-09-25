"""
Two-sided topic-share DID aligned with the revised sentiment/hate-speech DID.

The input is the already topic-labelled tweet-country data.  For every
origin-country -> mentioned-country pair and period, the script calculates the
share of observations assigned to each topic.  The denominator includes the
four substantive topics plus Other.

For each two-period comparison, the model is estimated with PanelOLS using
dyad fixed effects, period fixed effects, and standard errors clustered by
dyad.  The origin-side and mentioned-side log(1 + refugee inflow) interactions
enter the same regression.  For interpretation, estimates are converted after
estimation into sample-average discrete effects of increasing the original
rate by one applicant per 1,000 residents.  This positive rescaling leaves
t statistics and two-sided p values unchanged.
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    # Robustness estimation imports the data/model helpers but does not draw
    # the core script's figures.  Keep those helpers usable in a lean runtime.
    plt = None
    Patch = None
    FuncFormatter = None
    MaxNLocator = None
    MATPLOTLIB_AVAILABLE = False


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPIC_FILE = BUNDLE_ROOT / "data" / "cross_country_topic_labeled.xlsx"
DEFAULT_ASYLUM_FILE = BUNDLE_ROOT / "data" / "asylum_data.xlsx"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "results" / "topic_main"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT

MIN_COUNT_2015 = 20
MIN_COUNT_2022 = 30

# Treatment and reporting scales.  These match the revised sentiment and
# hate-speech DID code.
TREATMENT_TRANSFORM = "log1p"
REPORT_AVERAGE_DISCRETE_EFFECT = True
DISCRETE_INCREASE_PER_1000 = 1.0

# Topic shares are estimated on the 0-1 scale.  Multiply only the values sent
# to the figures by 100 so that the y axes report percentage-point changes.
# Model estimation and the numeric result files remain on their original scale.
FIGURE_PERCENTAGE_POINT_SCALE = 100.0

PRIMARY_TOPICS = [
    "Security",
    "Resource Pressure",
    "Governance",
    "Human Rights",
]
ALL_TOPICS = PRIMARY_TOPICS + ["Other"]
TOPIC_SLUGS = {
    "Security": "security",
    "Resource Pressure": "resource_pressure",
    "Governance": "governance",
    "Human Rights": "human_rights",
    "Other": "other",
}
SHARE_COLUMNS = {
    topic: f"share_{TOPIC_SLUGS[topic]}" for topic in ALL_TOPICS
}


# Publication-style figure settings aligned with the main DID code.
if MATPLOTLIB_AVAILABLE:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

TOPIC_COLORS = {
    "Governance": "#96C8A3",
    "Human Rights": "#8DBEDC",
    "Resource Pressure": "#F4BE7E",
    "Security": "#E99095",
}
ZERO_LINE_COLOR = "#A7A7A7"

# Match the current sentiment/hate-speech coefficient + dyad-distribution
# layout.  The left coefficient panel keeps the original visual weight and a
# compact, separately scaled box panel is added on the right.
ORIGINAL_MAIN_WIDTH = 5.5
BOX_PANEL_WIDTH = 0.58
FIG_W = 5.70
FIG_H = 5.15
MARKER_SIZE = 70
MARKER_EDGE_WIDTH = 1.9
ERROR_LINE_WIDTH = 2.2
BOX_LINE_WIDTH = ERROR_LINE_WIDTH
BOX_WIDTH = 0.30
SPINE_WIDTH = 1.5
TICK_SIZE = 15
LABEL_SIZE = 17
ANNOT_SIZE = 16
SHOW_YLABEL = True
SHOW_POST_LABEL = False
SAVE_PDF = False


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_country_name(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).strip()).lower()
    if value in {"", "nan", "none", "unknown"}:
        return np.nan
    return value


def normalize_topic(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).strip()).lower()
    mapping = {
        "security": "Security",
        "safety": "Security",
        "safety and security": "Security",
        "resource pressure": "Resource Pressure",
        "resource pressures": "Resource Pressure",
        "governance": "Governance",
        "human rights": "Human Rights",
        "human right": "Human Rights",
        "other": "Other",
    }
    return mapping.get(value, np.nan)


def check_input_files(topic_file: Path, asylum_file: Path) -> None:
    missing = [str(path) for path in [topic_file, asylum_file] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# Student-t distribution helpers (self-contained; SciPy is not required)
# ---------------------------------------------------------------------------
def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + m2) * (qap + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_value: float, df: int) -> float:
    if not np.isfinite(t_value) or df <= 0:
        return np.nan
    x = df / (df + t_value**2)
    return regularized_incomplete_beta(df / 2.0, 0.5, x)


def student_t_critical_975(df: int) -> float:
    if df <= 0:
        return np.nan
    low, high = 0.0, 20.0
    target_p = 0.05
    for _ in range(100):
        middle = (low + high) / 2.0
        if student_t_two_sided_p(middle, df) > target_p:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Topic-share panel construction
# ---------------------------------------------------------------------------
def read_topic_labeled_data(topic_file: Path, include_self: bool) -> pd.DataFrame:
    required = ["mentioned_country", "origin_country", "post_date", "topic_category"]
    frame = pd.read_excel(topic_file, usecols=required).copy()
    frame = frame.rename(
        columns={
            "mentioned_country": "mentioned",
            "origin_country": "origin",
            "post_date": "date",
            "topic_category": "topic",
        }
    )
    frame["origin"] = frame["origin"].map(clean_country_name)
    frame["mentioned"] = frame["mentioned"].map(clean_country_name)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["topic"] = frame["topic"].map(normalize_topic)
    frame = frame.dropna(subset=["origin", "mentioned", "date", "topic"]).copy()
    frame = frame.loc[frame["date"].dt.year.between(2010, 2024)].copy()
    if not include_self:
        frame = frame.loc[frame["origin"] != frame["mentioned"]].copy()
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["pair"] = frame["origin"] + "->" + frame["mentioned"]
    return frame[["pair", "origin", "mentioned", "year", "topic"]]


def build_topic_share_panel(
    topic_data: pd.DataFrame,
    year_to_period: dict[int, str],
    min_count: int,
    comparison_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    periods = list(dict.fromkeys(year_to_period.values()))
    frame = topic_data.loc[topic_data["year"].isin(year_to_period)].copy()
    frame["period"] = frame["year"].map(year_to_period)

    topic_counts = (
        frame.groupby(
            ["pair", "origin", "mentioned", "period", "topic"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "topic_count"})
    )
    pivot = topic_counts.pivot_table(
        index=["pair", "origin", "mentioned", "period"],
        columns="topic",
        values="topic_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None
    for topic in ALL_TOPICS:
        if topic not in pivot.columns:
            pivot[topic] = 0

    pivot["post_count"] = pivot[ALL_TOPICS].sum(axis=1).astype(int)
    for topic in ALL_TOPICS:
        pivot[SHARE_COLUMNS[topic]] = np.where(
            pivot["post_count"] > 0,
            pivot[topic] / pivot["post_count"],
            np.nan,
        )

    pairs_before_threshold = int(pivot["pair"].nunique())
    rows_before_threshold = int(len(pivot))
    pivot = pivot.loc[pivot["post_count"] >= min_count].copy()
    pairs_after_threshold = int(pivot["pair"].nunique())
    rows_after_threshold = int(len(pivot))

    period_counts = pivot.groupby("pair")["period"].nunique()
    common_pairs = period_counts.loc[period_counts == len(periods)].index
    panel = pivot.loc[pivot["pair"].isin(common_pairs)].copy()

    keep_columns = [
        "pair",
        "origin",
        "mentioned",
        "period",
        "post_count",
        *ALL_TOPICS,
        *[SHARE_COLUMNS[topic] for topic in ALL_TOPICS],
    ]
    panel = panel[keep_columns].sort_values(["pair", "period"])
    share_sum = panel[[SHARE_COLUMNS[t] for t in ALL_TOPICS]].sum(axis=1)
    diagnostics = {
        "comparison": comparison_name,
        "periods": " | ".join(periods),
        "min_count": min_count,
        "pairs_before_count_threshold": pairs_before_threshold,
        "rows_before_count_threshold": rows_before_threshold,
        "pairs_after_count_threshold": pairs_after_threshold,
        "rows_after_count_threshold": rows_after_threshold,
        "balanced_pairs_before_treatment": int(panel["pair"].nunique()),
        "balanced_rows_before_treatment": int(len(panel)),
        "max_abs_share_sum_minus_one": (
            float(np.max(np.abs(share_sum - 1.0))) if len(panel) else np.nan
        ),
    }
    return panel, diagnostics


# ---------------------------------------------------------------------------
# Treatment and two-period DID
# ---------------------------------------------------------------------------
def load_treatment(asylum_file: Path, year: int, treatment: str) -> pd.DataFrame:
    frame = pd.read_excel(asylum_file, sheet_name=str(year)).copy()
    required = ["host_country", treatment, "population"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"[asylum {year}] missing columns: {missing}")
    frame["country"] = frame["host_country"].map(clean_country_name)
    frame[treatment] = pd.to_numeric(frame[treatment], errors="coerce")
    frame["population"] = pd.to_numeric(frame["population"], errors="coerce")
    frame["raw_treatment_per_1000"] = np.where(
        frame["population"] > 0,
        1000.0 * frame[treatment] / frame["population"],
        np.nan,
    )
    if (frame["raw_treatment_per_1000"].dropna() < 0).any():
        raise ValueError(
            f"[asylum {year}] treatment {treatment!r} contains a negative rate"
        )

    if TREATMENT_TRANSFORM == "log1p":
        frame["treatment_model_scale"] = np.log1p(
            frame["raw_treatment_per_1000"]
        )
    elif TREATMENT_TRANSFORM == "raw":
        frame["treatment_model_scale"] = frame["raw_treatment_per_1000"]
    else:
        raise ValueError(
            "TREATMENT_TRANSFORM must be either 'log1p' or 'raw'; "
            f"received {TREATMENT_TRANSFORM!r}"
        )

    return frame[
        ["country", "raw_treatment_per_1000", "treatment_model_scale"]
    ].drop_duplicates("country")


def make_change_frame(
    panel: pd.DataFrame,
    treatment_frame: pd.DataFrame,
    baseline_period: str,
    comparison_period: str,
    crisis: str,
    treatment_name: str,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [SHARE_COLUMNS[topic] for topic in ALL_TOPICS]
    baseline = panel.loc[
        panel["period"] == baseline_period,
        ["pair", "origin", "mentioned", "post_count", *value_columns],
    ].copy()
    comparison = panel.loc[
        panel["period"] == comparison_period,
        ["pair", "origin", "mentioned", "post_count", *value_columns],
    ].copy()
    baseline = baseline.rename(
        columns={
            "post_count": "baseline_count",
            **{column: f"baseline_{column}" for column in value_columns},
        }
    )
    comparison = comparison.rename(
        columns={
            "post_count": "comparison_count",
            **{column: f"comparison_{column}" for column in value_columns},
        }
    )
    changes = baseline.merge(
        comparison,
        on=["pair", "origin", "mentioned"],
        how="inner",
        validate="one_to_one",
    )

    origin_treatment = treatment_frame.rename(
        columns={
            "country": "origin",
            "raw_treatment_per_1000": "raw_treatment_origin_per_1000",
            "treatment_model_scale": "treatment_origin_model_scale",
        }
    )
    mentioned_treatment = treatment_frame.rename(
        columns={
            "country": "mentioned",
            "raw_treatment_per_1000": "raw_treatment_mentioned_per_1000",
            "treatment_model_scale": "treatment_mentioned_model_scale",
        }
    )
    before_treatment = changes.copy()
    changes = changes.merge(origin_treatment, on="origin", how="left")
    changes = changes.merge(mentioned_treatment, on="mentioned", how="left")
    changes = changes.dropna(
        subset=[
            "raw_treatment_origin_per_1000",
            "raw_treatment_mentioned_per_1000",
            "treatment_origin_model_scale",
            "treatment_mentioned_model_scale",
        ]
    ).copy()

    for topic in ALL_TOPICS:
        share_column = SHARE_COLUMNS[topic]
        changes[f"delta_{share_column}"] = (
            changes[f"comparison_{share_column}"]
            - changes[f"baseline_{share_column}"]
        )
    changes["crisis"] = crisis
    changes["treatment"] = treatment_name
    changes["stage"] = stage
    changes["baseline_period"] = baseline_period
    changes["comparison_period"] = comparison_period

    matched_countries = set(treatment_frame.loc[
        treatment_frame["treatment_model_scale"].notna(), "country"
    ])
    unmatched_origin = sorted(set(before_treatment["origin"]) - matched_countries)
    unmatched_mentioned = sorted(
        set(before_treatment["mentioned"]) - matched_countries
    )
    match_rows = []
    for side, countries in [
        ("origin", unmatched_origin),
        ("mentioned", unmatched_mentioned),
    ]:
        for country in countries:
            match_rows.append(
                {
                    "crisis": crisis,
                    "treatment": treatment_name,
                    "stage": stage,
                    "side": side,
                    "unmatched_country": country,
                }
            )
    return changes, pd.DataFrame(match_rows)


def panel_ols_two_sided(
    baseline_outcome: np.ndarray,
    comparison_outcome: np.ndarray,
    treatment_origin: np.ndarray,
    treatment_mentioned: np.ndarray,
    pair_ids: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Estimate the two-sided DID with dyad and period fixed effects."""
    baseline_outcome = np.asarray(baseline_outcome, dtype=float)
    comparison_outcome = np.asarray(comparison_outcome, dtype=float)
    treatment_origin = np.asarray(treatment_origin, dtype=float)
    treatment_mentioned = np.asarray(treatment_mentioned, dtype=float)
    pair_ids = np.asarray(pair_ids, dtype=object)
    valid = (
        np.isfinite(baseline_outcome)
        & np.isfinite(comparison_outcome)
        & np.isfinite(treatment_origin)
        & np.isfinite(treatment_mentioned)
        & pd.notna(pair_ids)
    )
    y_baseline = baseline_outcome[valid]
    y_comparison = comparison_outcome[valid]
    x_origin = treatment_origin[valid]
    x_mentioned = treatment_mentioned[valid]
    pairs = pair_ids[valid].astype(str)
    n = len(pairs)

    # After removing dyad and period effects, identification comes from the
    # cross-dyad variation in the two treatment intensities.  Check that this
    # equivalent first-difference design has full rank before fitting PanelOLS.
    change_design = np.column_stack([np.ones(n), x_origin, x_mentioned])
    k = change_design.shape[1]

    empty_term = {
        "coef": np.nan,
        "se": np.nan,
        "t": np.nan,
        "p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
    }
    if n <= k or len(np.unique(pairs)) != n or np.linalg.matrix_rank(change_design) < k:
        model = {
            "n_pairs": int(n),
            "df_resid": int(max(n - k, 0)),
            "r_squared": np.nan,
            "condition_number": np.nan,
        }
        return {"origin": empty_term.copy(), "mentioned": empty_term.copy()}, model

    baseline = pd.DataFrame(
        {
            "pair": pairs,
            "period_id": 0,
            "Y": y_baseline,
            "DID_origin": 0.0,
            "DID_mentioned": 0.0,
        }
    )
    comparison = pd.DataFrame(
        {
            "pair": pairs,
            "period_id": 1,
            "Y": y_comparison,
            "DID_origin": x_origin,
            "DID_mentioned": x_mentioned,
        }
    )
    panel = (
        pd.concat([baseline, comparison], ignore_index=True)
        .set_index(["pair", "period_id"])
        .sort_index()
    )

    fitted = PanelOLS(
        dependent=panel["Y"],
        exog=panel[["DID_origin", "DID_mentioned"]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        cluster_entity=True,
    )

    confidence = fitted.conf_int()

    terms = {}
    for side, term_name in [
        ("origin", "DID_origin"),
        ("mentioned", "DID_mentioned"),
    ]:
        terms[side] = {
            "coef": float(fitted.params[term_name]),
            "se": float(fitted.std_errors[term_name]),
            "t": float(fitted.tstats[term_name]),
            "p": float(fitted.pvalues[term_name]),
            "ci_low": float(confidence.loc[term_name].iloc[0]),
            "ci_high": float(confidence.loc[term_name].iloc[1]),
        }

    model = {
        "n_pairs": int(n),
        "df_resid": int(fitted.df_resid),
        "r_squared": float(fitted.rsquared),
        "condition_number": float(np.linalg.cond(change_design)),
    }
    return terms, model


def average_discrete_factor(raw_treatment: np.ndarray) -> float:
    """Mean model-scale change when the raw rate rises by the chosen increment."""
    raw_treatment = np.asarray(raw_treatment, dtype=float)
    raw_treatment = raw_treatment[np.isfinite(raw_treatment)]
    if len(raw_treatment) == 0:
        return np.nan
    if (raw_treatment < 0).any():
        raise ValueError("Raw treatment rates must be non-negative")

    increment = float(DISCRETE_INCREASE_PER_1000)
    if increment <= 0:
        raise ValueError("DISCRETE_INCREASE_PER_1000 must be positive")
    if TREATMENT_TRANSFORM == "log1p":
        model_scale_changes = (
            np.log1p(raw_treatment + increment) - np.log1p(raw_treatment)
        )
    elif TREATMENT_TRANSFORM == "raw":
        model_scale_changes = np.full(len(raw_treatment), increment, dtype=float)
    else:
        raise ValueError(
            "TREATMENT_TRANSFORM must be either 'log1p' or 'raw'; "
            f"received {TREATMENT_TRANSFORM!r}"
        )
    return float(np.mean(model_scale_changes))


def convert_to_average_discrete_effect(
    term: dict[str, float], raw_treatment: np.ndarray
) -> dict[str, float]:
    """Keep the fitted log-scale result and add the interpretable AMDE result."""
    result = dict(term)
    result.update(
        {
            "model_coef": term["coef"],
            "model_se": term["se"],
            "model_ci_low": term["ci_low"],
            "model_ci_high": term["ci_high"],
        }
    )
    factor = average_discrete_factor(raw_treatment)
    result["average_discrete_factor"] = factor

    if REPORT_AVERAGE_DISCRETE_EFFECT:
        for name in ["coef", "se", "ci_low", "ci_high"]:
            result[name] = (
                float(term[name]) * factor
                if pd.notna(term[name]) and pd.notna(factor)
                else np.nan
            )
    return result


def estimate_all_topics(
    changes: pd.DataFrame,
    crisis: str,
    treatment_name: str,
    stage: str,
    baseline_period: str,
    comparison_period: str,
) -> pd.DataFrame:
    rows = []
    for topic in ALL_TOPICS:
        share_column = SHARE_COLUMNS[topic]
        terms, model = panel_ols_two_sided(
            changes[f"baseline_{share_column}"].to_numpy(),
            changes[f"comparison_{share_column}"].to_numpy(),
            changes["treatment_origin_model_scale"].to_numpy(),
            changes["treatment_mentioned_model_scale"].to_numpy(),
            changes["pair"].to_numpy(),
        )
        for side in ["origin", "mentioned"]:
            reported_term = convert_to_average_discrete_effect(
                terms[side],
                changes[f"raw_treatment_{side}_per_1000"].to_numpy(),
            )
            rows.append(
                {
                    "crisis": crisis,
                    "treatment": treatment_name,
                    "stage": stage,
                    "topic": topic,
                    "primary_topic": topic in PRIMARY_TOPICS,
                    "side": side,
                    "baseline_period": baseline_period,
                    "comparison_period": comparison_period,
                    **reported_term,
                    **model,
                    "n_origins": int(changes["origin"].nunique()),
                    "n_mentioned": int(changes["mentioned"].nunique()),
                    "outcome_unit": "topic share (0-1)",
                    "model_treatment_scale": (
                        "log(1 + applicants per 1,000 residents)"
                        if TREATMENT_TRANSFORM == "log1p"
                        else "applicants per 1,000 residents"
                    ),
                    "treatment_unit": "applicants per 1,000 residents",
                    "reported_effect_scale": (
                        "sample-average discrete effect of +1 applicant "
                        "per 1,000 residents"
                        if REPORT_AVERAGE_DISCRETE_EFFECT
                        else "model coefficient"
                    ),
                    "covariance": "dyad-clustered standard errors",
                }
            )
    return pd.DataFrame(rows)


def run_comparison(
    panel: pd.DataFrame,
    treatment_frame: pd.DataFrame,
    baseline_period: str,
    comparison_period: str,
    crisis: str,
    treatment_name: str,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    changes, match_diagnostics = make_change_frame(
        panel,
        treatment_frame,
        baseline_period,
        comparison_period,
        crisis,
        treatment_name,
        stage,
    )
    results = estimate_all_topics(
        changes,
        crisis,
        treatment_name,
        stage,
        baseline_period,
        comparison_period,
    )
    sample = {
        "crisis": crisis,
        "treatment": treatment_name,
        "stage": stage,
        "baseline_period": baseline_period,
        "comparison_period": comparison_period,
        "balanced_pairs_before_treatment": int(panel["pair"].nunique()),
        "pairs_after_both_treatments_matched": int(changes["pair"].nunique()),
        "panel_rows_after_both_treatments_matched": int(2 * len(changes)),
        "n_origins": int(changes["origin"].nunique()),
        "n_mentioned": int(changes["mentioned"].nunique()),
    }
    return results, changes, match_diagnostics, sample


def build_main_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = results.loc[results["primary_topic"]].copy()
    for (crisis, treatment, topic), group in primary.groupby(
        ["crisis", "treatment", "topic"], sort=False
    ):
        row = {"crisis": crisis, "treatment": treatment, "topic": topic}
        for side in ["origin", "mentioned"]:
            pre = group.loc[(group["stage"] == "pretrend") & (group["side"] == side)]
            main = group.loc[(group["stage"] == "main_did") & (group["side"] == side)]
            pre_row = pre.iloc[0] if not pre.empty else None
            main_row = main.iloc[0] if not main.empty else None
            for prefix, source in [("pretrend", pre_row), ("main_did", main_row)]:
                for statistic in [
                    "coef",
                    "se",
                    "t",
                    "p",
                    "ci_low",
                    "ci_high",
                    "model_coef",
                    "model_se",
                    "model_ci_low",
                    "model_ci_high",
                    "average_discrete_factor",
                    "n_pairs",
                    "n_origins",
                    "n_mentioned",
                ]:
                    row[f"{side}_{prefix}_{statistic}"] = (
                        source[statistic] if source is not None else np.nan
                    )
            row[f"{side}_parallel_trend_pass"] = (
                bool(pre_row["p"] >= 0.05)
                if pre_row is not None and pd.notna(pre_row["p"])
                else False
            )
            row[f"{side}_main_did_significant"] = (
                bool(main_row["p"] < 0.05)
                if main_row is not None and pd.notna(main_row["p"])
                else False
            )
            row[f"{side}_final_significant"] = bool(
                row[f"{side}_parallel_trend_pass"]
                and row[f"{side}_main_did_significant"]
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    topic_order = {topic: index for index, topic in enumerate(PRIMARY_TOPICS)}
    summary["_topic_order"] = summary["topic"].map(topic_order)
    return summary.sort_values(
        ["crisis", "treatment", "_topic_order"]
    ).drop(columns="_topic_order")


def build_compositional_qc(
    results: pd.DataFrame, panels: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    coefficient_qc = (
        results.groupby(["crisis", "treatment", "stage", "side"], as_index=False)
        .agg(
            sum_of_five_topic_coefficients=("coef", "sum"),
            sum_of_five_model_coefficients=("model_coef", "sum"),
            max_condition_number=("condition_number", "max"),
            n_pairs=("n_pairs", "max"),
        )
    )
    panel_errors = []
    for comparison, panel in panels.items():
        share_sum = panel[[SHARE_COLUMNS[t] for t in ALL_TOPICS]].sum(axis=1)
        panel_errors.append(
            {
                "comparison": comparison,
                "max_abs_share_sum_minus_one": float(
                    np.max(np.abs(share_sum - 1.0))
                ),
                "n_panel_rows": int(len(panel)),
                "n_pairs": int(panel["pair"].nunique()),
            }
        )
    coefficient_qc["absolute_coefficient_sum"] = coefficient_qc[
        "sum_of_five_topic_coefficients"
    ].abs()
    coefficient_qc["absolute_model_coefficient_sum"] = coefficient_qc[
        "sum_of_five_model_coefficients"
    ].abs()
    return coefficient_qc, pd.DataFrame(panel_errors)


def sanitize_filename(value) -> str:
    """Keep readable names while replacing characters invalid on Windows."""
    text = str(value)
    for character in '/\\:*?"<>|':
        text = text.replace(character, "_")
    return text


def format_coef_label(value, ci_low=None, ci_high=None) -> str:
    del ci_low, ci_high
    if pd.isna(value):
        return ""
    return f"Effect = {value:.2f} percentage points"


def dynamic_formatter(value, position) -> str:
    del position
    if np.isclose(value, 0, atol=1e-12):
        return "0.00"
    if abs(value) >= 1:
        return f"{value:.1f}"
    if abs(value) >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def nice_ylim_raw(ci_low: list[float], ci_high: list[float]) -> tuple[float, float]:
    values = [value for value in ci_low + ci_high if pd.notna(value)]
    if not values:
        return -5.0, 5.0
    limit = max(abs(min(values)), abs(max(values)), 0.0) * 1.22
    limit = max(limit, 3.0)
    return -limit, limit


def nice_ylim_distribution(values: list[float]) -> tuple[float, float]:
    """Return a tight, non-symmetric range for a dyad distribution."""
    finite = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if len(finite) == 0:
        return -5.0, 5.0
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    if span <= 1e-12:
        span = max(abs(lower) * 0.20, 1.0)
    padding = 0.18 * span
    return lower - padding, upper + padding


def ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    if pd.isna(ci_low) or pd.isna(ci_high):
        return False
    return bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))


def dyad_specific_discrete_effects(
    changes: pd.DataFrame,
    row: pd.Series,
    side: str,
    stage: str,
) -> pd.DataFrame:
    """Return the reported discrete effect for every dyad in one model."""
    if side not in {"origin", "mentioned"}:
        raise ValueError(f"Unknown side: {side}")
    if stage not in {"pretrend", "main_did"}:
        raise ValueError(f"Unknown stage: {stage}")

    subset = changes.loc[
        (changes["crisis"].astype(str) == str(row["crisis"]))
        & (changes["treatment"].astype(str) == str(row["treatment"]))
        & (changes["stage"].astype(str) == stage)
    ].copy()
    raw_column = f"raw_treatment_{side}_per_1000"
    required = ["pair", "origin", "mentioned", raw_column]
    missing = [column for column in required if column not in subset.columns]
    if missing:
        raise ValueError(f"Dyad-effect input is missing columns: {missing}")

    data = subset[required].dropna().copy()
    data[raw_column] = pd.to_numeric(data[raw_column], errors="coerce")
    data = data.dropna(subset=[raw_column]).copy()
    variation = data.groupby("pair")[raw_column].nunique(dropna=False)
    if (variation > 1).any():
        bad = variation.loc[variation > 1].index.tolist()[:5]
        raise ValueError(
            f"Treatment varies within dyad for {raw_column}; examples: {bad}"
        )
    data = data.drop_duplicates("pair").sort_values("pair").reset_index(drop=True)
    if (data[raw_column] < 0).any():
        raise ValueError(f"Negative treatment rate in {raw_column}")

    model_coefficient = float(row[f"{side}_{stage}_model_coef"])
    increment = float(DISCRETE_INCREASE_PER_1000)
    if TREATMENT_TRANSFORM == "log1p":
        contrast = (
            np.log1p(data[raw_column] + increment)
            - np.log1p(data[raw_column])
        )
    elif TREATMENT_TRANSFORM == "raw":
        contrast = np.full(len(data), increment, dtype=float)
    else:
        raise ValueError(f"Unknown treatment transform: {TREATMENT_TRANSFORM}")

    data = data.rename(columns={raw_column: "raw_treatment_per_1000"})
    data["model_scale_contrast"] = contrast
    data["dyad_discrete_effect"] = model_coefficient * contrast
    data["crisis"] = str(row["crisis"])
    data["treatment"] = str(row["treatment"])
    data["topic"] = str(row["topic"])
    data["side"] = side
    data["stage"] = stage

    reported = float(row[f"{side}_{stage}_coef"])
    dyad_mean = float(data["dyad_discrete_effect"].mean())
    if not np.isclose(dyad_mean, reported, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "Dyad mean does not match the reported average discrete effect: "
            f"mean={dyad_mean}, reported={reported}"
        )
    return data


def dyad_effects_for_figure(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert dyad-specific topic-share effects to percentage points."""
    plotted = frame.copy()
    plotted["dyad_discrete_effect"] = (
        pd.to_numeric(plotted["dyad_discrete_effect"], errors="coerce")
        * FIGURE_PERCENTAGE_POINT_SCALE
    )
    return plotted


def draw_dyad_distribution(axis, x_value: float, frame: pd.DataFrame, color: str):
    """Draw Q25-Q75, min-max whiskers, and a black dyad-mean line."""
    values = frame["dyad_discrete_effect"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None
    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    mean = float(np.mean(values))
    axis.bxp(
        [{
            "label": "",
            "whislo": minimum,
            "q1": float(q25),
            "med": mean,
            "q3": float(q75),
            "whishi": maximum,
            "fliers": [],
        }],
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
        medianprops={"color": "none", "linewidth": 0},
        whiskerprops={"color": color, "linewidth": BOX_LINE_WIDTH},
        capprops={"color": color, "linewidth": BOX_LINE_WIDTH},
        zorder=3,
    )
    mean_half_width = BOX_WIDTH * 0.50
    axis.plot(
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


def style_axis(axis) -> None:
    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
        color=ZERO_LINE_COLOR,
        zorder=1,
    )
    axis.set_xlim(-0.30, 2.55)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
    axis.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        width=0.8,
        length=3.2,
        top=False,
        right=False,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(SPINE_WIDTH)
    axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.grid(False)


def draw_coef_point(
    axis,
    x_value: float,
    y_value: float,
    ci_low: float,
    ci_high: float,
    color: str,
) -> None:
    if pd.isna(y_value):
        return
    if pd.notna(ci_low) and pd.notna(ci_high):
        axis.errorbar(
            [x_value],
            [y_value],
            yerr=[[y_value - ci_low], [ci_high - y_value]],
            fmt="none",
            ecolor=color,
            elinewidth=ERROR_LINE_WIDTH,
            capsize=0,
            alpha=1.0,
            zorder=4,
        )
    axis.scatter(
        [x_value],
        [y_value],
        s=MARKER_SIZE,
        facecolors=color if ci_excludes_zero(ci_low, ci_high) else "white",
        edgecolors=color,
        linewidths=MARKER_EDGE_WIDTH,
        alpha=1.0,
        zorder=5,
    )


def period_labels(crisis: str) -> list[str]:
    if str(crisis) == "2015":
        return [
            "2010-2012\n(pre 1)",
            "2013-2014\n(baseline)",
            "2015-2021\n(post 1)",
        ]
    return [
        "2010-2014\n(pre 1)",
        "2015-2021\n(baseline)",
        "2022-2024\n(post 1)",
    ]


def treatment_filename_label(treatment: str) -> str:
    if treatment == "non-Ukrainian_first_time_applicant":
        return "non-Ukrainian first-time asylum applicant"
    if treatment == "first_time_applicants":
        return "first-time asylum applicant"
    if treatment == "first_time_applicants_ukrainian":
        return "first-time asylum applicant (Ukrainians)"
    return str(treatment)


def figure_filename(row: pd.Series, side: str, extension: str = "png") -> str:
    crisis_prefix = "crisis2015" if str(row["crisis"]) == "2015" else "2022"
    parts = [
        crisis_prefix,
        str(row["topic"]).lower(),
        treatment_filename_label(str(row["treatment"])),
        side,
    ]
    return sanitize_filename("_".join(parts) + f".{extension}")


def draw_single_topic_plot(
    row: pd.Series,
    side: str,
    all_changes: pd.DataFrame,
    figure_dir: Path,
) -> list[str]:
    x_values = [0, 1, 2]
    y_values = [
        FIGURE_PERCENTAGE_POINT_SCALE
        * float(row[f"{side}_pretrend_coef"]),
        0.0,
        FIGURE_PERCENTAGE_POINT_SCALE
        * float(row[f"{side}_main_did_coef"]),
    ]
    ci_low = [
        FIGURE_PERCENTAGE_POINT_SCALE
        * float(row[f"{side}_pretrend_ci_low"]),
        0.0,
        FIGURE_PERCENTAGE_POINT_SCALE
        * float(row[f"{side}_main_did_ci_low"]),
    ]
    ci_high = [
        FIGURE_PERCENTAGE_POINT_SCALE
        * float(row[f"{side}_pretrend_ci_high"]),
        0.0,
        FIGURE_PERCENTAGE_POINT_SCALE
        * float(row[f"{side}_main_did_ci_high"]),
    ]
    color = TOPIC_COLORS[str(row["topic"])]
    post_dyads = dyad_effects_for_figure(
        dyad_specific_discrete_effects(
            all_changes,
            row,
            side,
            "main_did",
        )
    )

    figure, (axis, box_axis) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={
            "width_ratios": [ORIGINAL_MAIN_WIDTH, BOX_PANEL_WIDTH],
            "wspace": 0.085,
        },
    )
    # The middle period is the omitted reference category, not an estimated
    # zero-effect coefficient; label it on the x axis but do not draw a point.
    for index in (0, 2):
        draw_coef_point(
            axis,
            x_values[index],
            y_values[index],
            ci_low[index],
            ci_high[index],
            color,
        )

    axis.set_ylim(*nice_ylim_raw(ci_low, ci_high))
    style_axis(axis)
    axis.set_xticks(x_values)
    axis.set_xticklabels(period_labels(str(row["crisis"])))
    axis.set_xlabel("")
    if SHOW_YLABEL:
        axis.set_ylabel(
            "Average change in topic share\n(percentage points)",
            fontsize=LABEL_SIZE,
            labelpad=9,
        )
    if SHOW_POST_LABEL and pd.notna(y_values[2]):
        axis.annotate(
            format_coef_label(y_values[2], ci_low[2], ci_high[2]),
            xy=(x_values[2], y_values[2]),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=ANNOT_SIZE,
            color=color,
            fontweight="bold",
            zorder=6,
        )

    # The right panel follows the current sentiment figure: Q25-Q75 box,
    # min-max whiskers, and a black mean line, all on a separate tight scale.
    draw_dyad_distribution(box_axis, 0.0, post_dyads, color)
    box_values = post_dyads["dyad_discrete_effect"].dropna().tolist()
    box_ymin, box_ymax = nice_ylim_distribution(box_values)
    box_axis.set_ylim(box_ymin, box_ymax)
    box_axis.set_xlim(-0.46, 0.46)
    box_axis.set_xticks([0.0])
    post_period = (
        "2015–2021" if str(row["crisis"]) == "2015" else "2022–2024"
    )
    box_axis.set_xticklabels([post_period])
    box_axis.set_xlabel("")
    box_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    box_axis.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
    box_axis.yaxis.tick_right()
    box_axis.yaxis.set_label_position("right")
    box_axis.tick_params(
        axis="both",
        labelsize=max(TICK_SIZE - 1, 8),
        width=0.8,
        length=3.0,
        top=False,
        left=False,
        right=True,
    )
    box_axis.spines["top"].set_visible(False)
    box_axis.spines["left"].set_visible(False)
    box_axis.spines["right"].set_linewidth(SPINE_WIDTH)
    box_axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
    box_axis.spines["right"].set_color("#333333")
    box_axis.spines["bottom"].set_color("#333333")
    box_axis.set_facecolor("white")
    box_axis.set_title("")
    box_axis.set_ylabel(
        "Dyad estimates\n(percentage points)",
        fontsize=max(LABEL_SIZE - 1, 9),
        labelpad=8,
    )
    if box_ymin <= 0 <= box_ymax:
        box_axis.axhline(
            0,
            linestyle="--",
            linewidth=0.9,
            color=ZERO_LINE_COLOR,
            zorder=1,
        )

    figure.tight_layout()
    png_path = figure_dir / figure_filename(row, side, "png")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    created = [str(png_path)]
    if SAVE_PDF:
        pdf_path = figure_dir / figure_filename(row, side, "pdf")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        created.append(str(pdf_path))
    plt.close(figure)
    return created


def draw_combined_2022_total_figure(
    summary: pd.DataFrame,
    all_changes: pd.DataFrame,
    figure_dir: Path,
    reference_png: Path | None = None,
) -> Path:
    """Combine the two significant 2022 total-inflow mentioned-side results."""
    specifications = [
        ("Governance", -0.055, -0.18),
        ("Human Rights", 0.055, 0.18),
    ]
    selected = []
    for topic, coefficient_offset, box_position in specifications:
        rows = summary.loc[
            (summary["crisis"].astype(str) == "2022")
            & (summary["treatment"].astype(str) == "non-Ukrainian_first_time_applicant")
            & (summary["topic"].astype(str) == topic)
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one 2022 non-Ukrainian_first_time_applicant row for {topic}; found {len(rows)}"
            )
        row = rows.iloc[0]
        if not bool(row["mentioned_final_significant"]):
            raise RuntimeError(
                f"The requested combined result is not final-significant: {topic}"
            )
        dyads = dyad_specific_discrete_effects(
            all_changes,
            row,
            "mentioned",
            "main_did",
        )
        dyads = dyad_effects_for_figure(dyads)
        selected.append((topic, row, coefficient_offset, box_position, dyads))

    figure, (axis, box_axis) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={
            "width_ratios": [ORIGINAL_MAIN_WIDTH, BOX_PANEL_WIDTH],
            "wspace": 0.085,
        },
    )
    x_values = np.array([0.0, 1.0, 2.0])
    coefficient_scale_values = []
    box_scale_values = []

    for topic, row, coefficient_offset, box_position, dyads in selected:
        color = TOPIC_COLORS[topic]
        values = [
            FIGURE_PERCENTAGE_POINT_SCALE
            * float(row["mentioned_pretrend_coef"]),
            0.0,
            FIGURE_PERCENTAGE_POINT_SCALE
            * float(row["mentioned_main_did_coef"]),
        ]
        lows = [
            FIGURE_PERCENTAGE_POINT_SCALE
            * float(row["mentioned_pretrend_ci_low"]),
            0.0,
            FIGURE_PERCENTAGE_POINT_SCALE
            * float(row["mentioned_main_did_ci_low"]),
        ]
        highs = [
            FIGURE_PERCENTAGE_POINT_SCALE
            * float(row["mentioned_pretrend_ci_high"]),
            0.0,
            FIGURE_PERCENTAGE_POINT_SCALE
            * float(row["mentioned_main_did_ci_high"]),
        ]
        for index in (0, 2):
            draw_coef_point(
                axis,
                float(x_values[index] + coefficient_offset),
                values[index],
                lows[index],
                highs[index],
                color,
            )
        coefficient_scale_values.extend(values + lows + highs)
        draw_dyad_distribution(box_axis, box_position, dyads, color)
        box_scale_values.extend(
            dyads["dyad_discrete_effect"].dropna().astype(float).tolist()
        )

    coefficient_scale_values = [
        value for value in coefficient_scale_values if pd.notna(value)
    ]
    axis.set_ylim(
        *nice_ylim_raw(coefficient_scale_values, coefficient_scale_values)
    )
    style_axis(axis)
    axis.set_xticks(x_values)
    axis.set_xticklabels(period_labels("2022"))
    axis.set_xlabel("")
    axis.set_ylabel(
        "Average change in topic share\n(percentage points)",
        fontsize=LABEL_SIZE,
        labelpad=9,
    )

    box_ymin, box_ymax = nice_ylim_distribution(box_scale_values)
    box_axis.set_ylim(box_ymin, box_ymax)
    box_axis.set_xlim(-0.46, 0.46)
    box_axis.set_xticks([0.0])
    box_axis.set_xticklabels(["2022–2024"])
    box_axis.set_xlabel("")
    box_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    box_axis.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
    box_axis.yaxis.tick_right()
    box_axis.yaxis.set_label_position("right")
    box_axis.tick_params(
        axis="both",
        labelsize=max(TICK_SIZE - 1, 8),
        width=0.8,
        length=3.0,
        top=False,
        left=False,
        right=True,
    )
    box_axis.spines["top"].set_visible(False)
    box_axis.spines["left"].set_visible(False)
    box_axis.spines["right"].set_linewidth(SPINE_WIDTH)
    box_axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
    box_axis.spines["right"].set_color("#333333")
    box_axis.spines["bottom"].set_color("#333333")
    box_axis.set_ylabel(
        "Dyad estimates\n(percentage points)",
        fontsize=max(LABEL_SIZE - 1, 9),
        labelpad=8,
    )
    box_axis.set_title("")
    if box_ymin <= 0 <= box_ymax:
        box_axis.axhline(
            0,
            linestyle="--",
            linewidth=0.9,
            color=ZERO_LINE_COLOR,
            zorder=1,
        )

    figure.tight_layout()
    output_path = (
        figure_dir
        / "2022_non_ukrainian_first_time_asylum_applicants_two_significant_topics.png"
    )
    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # bbox_inches="tight" can make the exported PNG dimensions differ slightly
    # even when figsize is identical.  Match this one extra combined figure to
    # the actual pixel dimensions of a regular topic figure, without changing
    # any model, axis, colour, or annotation settings.
    if reference_png is not None and reference_png.exists():
        from PIL import Image

        with Image.open(reference_png) as reference_image:
            target_width, target_height = reference_image.size

        with Image.open(output_path) as source_image:
            source_image = source_image.convert("RGB")
            source_width, source_height = source_image.size

            if (source_width, source_height) != (target_width, target_height):
                scale = min(
                    target_width / source_width,
                    target_height / source_height,
                )
                resized_width = max(1, round(source_width * scale))
                resized_height = max(1, round(source_height * scale))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                resized = source_image.resize(
                    (resized_width, resized_height),
                    resampling,
                )
                canvas = Image.new(
                    "RGB",
                    (target_width, target_height),
                    "white",
                )
                left = (target_width - resized_width) // 2
                top = (target_height - resized_height) // 2
                canvas.paste(resized, (left, top))
                canvas.save(output_path, dpi=(600, 600))

    return output_path


def draw_topic_color_legend(figure_dir: Path) -> Path:
    """Save the four-topic colour key used across all topic figures."""
    figure, axis = plt.subplots(figsize=(10.2, 0.72))
    axis.axis("off")
    handles = [
        Patch(facecolor=TOPIC_COLORS[topic], edgecolor="none", label=topic)
        for topic in ["Governance", "Human Rights", "Resource Pressure", "Security"]
    ]
    axis.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=15,
        columnspacing=1.35,
        handlelength=2.1,
        handletextpad=0.28,
    )
    output_path = figure_dir / "topic_color_legend.png"
    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_results(
    summary: pd.DataFrame,
    all_changes: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Draw sentiment-style coefficient and dyad-distribution topic figures."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for _, row in summary.iterrows():
        for side in ["origin", "mentioned"]:
            created.extend(
                draw_single_topic_plot(row, side, all_changes, figure_dir)
            )
    reference_png = next(
        (Path(path) for path in created if Path(path).suffix.lower() == ".png"),
        None,
    )
    created.append(
        str(
            draw_combined_2022_total_figure(
                summary,
                all_changes,
                figure_dir,
                reference_png=reference_png,
            )
        )
    )
    created.append(str(draw_topic_color_legend(figure_dir)))

    # Export the exact dyad values represented by the post-period box plots.
    dyad_parts = []
    for _, row in summary.iterrows():
        for side in ["origin", "mentioned"]:
            dyad_parts.append(
                dyad_specific_discrete_effects(
                    all_changes,
                    row,
                    side,
                    "main_did",
                )
            )
    pd.concat(dyad_parts, ignore_index=True).to_csv(
        output_dir / "topic_share_post_dyad_specific_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return created


def save_outputs(
    output_dir: Path,
    raw_results: pd.DataFrame,
    summary: pd.DataFrame,
    changes: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    construction_diagnostics: list[dict[str, object]],
    sample_diagnostics: list[dict[str, object]],
    match_diagnostics: pd.DataFrame,
    coefficient_qc: pd.DataFrame,
    panel_qc: pd.DataFrame,
    settings: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results.to_csv(
        output_dir / "topic_share_all_results.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output_dir / "topic_share_main_summary.csv", index=False, encoding="utf-8-sig"
    )
    changes.to_csv(
        output_dir / "topic_share_pair_changes.csv", index=False, encoding="utf-8-sig"
    )
    for name, panel in panels.items():
        panel.to_csv(
            output_dir / f"topic_share_panel_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    pd.DataFrame(construction_diagnostics).to_csv(
        output_dir / "topic_share_panel_construction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(sample_diagnostics).to_csv(
        output_dir / "topic_share_sample_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    match_diagnostics.to_csv(
        output_dir / "topic_share_unmatched_countries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coefficient_qc.to_csv(
        output_dir / "topic_share_compositional_coefficient_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel_qc.to_csv(
        output_dir / "topic_share_compositional_panel_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([settings]).to_csv(
        output_dir / "topic_share_run_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    warnings = []
    for sample in sample_diagnostics:
        n_pairs = sample["pairs_after_both_treatments_matched"]
        if n_pairs < 30:
            warnings.append(
                f"{sample['crisis']} / {sample['treatment']} / {sample['stage']}: "
                f"only {n_pairs} balanced country pairs after treatment matching."
            )
    failed = summary[
        (~summary["origin_parallel_trend_pass"])
        | (~summary["mentioned_parallel_trend_pass"])
    ]
    if not failed.empty:
        warnings.append(
            "At least one pre-trend test fails for the specifications "
            "listed in topic_share_main_summary.csv."
        )
    (output_dir / "ANALYSIS_WARNINGS.txt").write_text(
        "\n".join(warnings) if warnings else "No automatic warnings.",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-file", type=Path, default=DEFAULT_TOPIC_FILE)
    parser.add_argument("--asylum-file", type=Path, default=DEFAULT_ASYLUM_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-count-2015", type=int, default=MIN_COUNT_2015)
    parser.add_argument("--min-count-2022", type=int, default=MIN_COUNT_2022)
    parser.add_argument("--include-self", action="store_true")
    # Jupyter/IPython automatically appends arguments such as
    # ``-f .../kernel-<id>.json``.  They are unrelated to this analysis and
    # would make parse_args() terminate the notebook with SystemExit: 2.
    # parse_known_args() keeps all of this script's real command-line options
    # while safely ignoring those notebook-kernel arguments.
    args, _notebook_arguments = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )
    check_input_files(args.topic_file, args.asylum_file)

    print("Reading topic-labelled data...")
    topic_data = read_topic_labeled_data(args.topic_file, args.include_self)
    print(
        f"Usable labelled rows: {len(topic_data):,}; "
        f"origins={topic_data['origin'].nunique()}; "
        f"mentioned={topic_data['mentioned'].nunique()}"
    )

    panel_2015_pre, construction_2015_pre = build_topic_share_panel(
        topic_data,
        {
            2010: "2010-2012",
            2011: "2010-2012",
            2012: "2010-2012",
            2013: "2013-2014",
            2014: "2013-2014",
        },
        args.min_count_2015,
        "2015_pretrend",
    )
    panel_2015_main, construction_2015_main = build_topic_share_panel(
        topic_data,
        {
            2013: "2013-2014",
            2014: "2013-2014",
            **{year: "2015-2021" for year in range(2015, 2022)},
        },
        args.min_count_2015,
        "2015_main",
    )
    panel_2022_pre, construction_2022_pre = build_topic_share_panel(
        topic_data,
        {
            **{year: "2010-2014" for year in range(2010, 2015)},
            **{year: "2015-2021" for year in range(2015, 2022)},
        },
        args.min_count_2022,
        "2022_pretrend",
    )
    panel_2022_main, construction_2022_main = build_topic_share_panel(
        topic_data,
        {
            **{year: "2015-2021" for year in range(2015, 2022)},
            **{year: "2022-2024" for year in range(2022, 2025)},
        },
        args.min_count_2022,
        "2022_main",
    )
    panels = {
        "2015_pretrend": panel_2015_pre,
        "2015_main": panel_2015_main,
        "2022_pretrend": panel_2022_pre,
        "2022_main": panel_2022_main,
    }
    construction_diagnostics = [
        construction_2015_pre,
        construction_2015_main,
        construction_2022_pre,
        construction_2022_main,
    ]

    comparison_specs = []
    treatment_2015 = load_treatment(args.asylum_file, 2015, "first_time_applicants")
    comparison_specs.extend(
        [
            (
                panel_2015_pre,
                treatment_2015,
                "2013-2014",
                "2010-2012",
                "2015",
                "first_time_applicants",
                "pretrend",
            ),
            (
                panel_2015_main,
                treatment_2015,
                "2013-2014",
                "2015-2021",
                "2015",
                "first_time_applicants",
                "main_did",
            ),
        ]
    )
    for treatment_name in ["non-Ukrainian_first_time_applicant", "first_time_applicants_ukrainian"]:
        treatment_2022 = load_treatment(
            args.asylum_file, 2022, treatment_name
        )
        comparison_specs.extend(
            [
                (
                    panel_2022_pre,
                    treatment_2022,
                    "2015-2021",
                    "2010-2014",
                    "2022",
                    treatment_name,
                    "pretrend",
                ),
                (
                    panel_2022_main,
                    treatment_2022,
                    "2015-2021",
                    "2022-2024",
                    "2022",
                    treatment_name,
                    "main_did",
                ),
            ]
        )

    result_parts = []
    change_parts = []
    match_parts = []
    sample_diagnostics = []
    for spec in comparison_specs:
        result, changes, match, sample = run_comparison(*spec)
        result_parts.append(result)
        change_parts.append(changes)
        if not match.empty:
            match_parts.append(match)
        sample_diagnostics.append(sample)

    raw_results = pd.concat(result_parts, ignore_index=True)
    summary = build_main_summary(raw_results)
    all_changes = pd.concat(change_parts, ignore_index=True)
    match_diagnostics = (
        pd.concat(match_parts, ignore_index=True)
        if match_parts
        else pd.DataFrame(
            columns=["crisis", "treatment", "stage", "side", "unmatched_country"]
        )
    )
    coefficient_qc, panel_qc = build_compositional_qc(raw_results, panels)

    settings = {
        "topic_file": str(args.topic_file),
        "asylum_file": str(args.asylum_file),
        "include_self": args.include_self,
        "min_count_2015": args.min_count_2015,
        "min_count_2022": args.min_count_2022,
        "primary_topics": " | ".join(PRIMARY_TOPICS),
        "denominator_topics": " | ".join(ALL_TOPICS),
        "model": (
            "PanelOLS with dyad and period fixed effects; "
            "origin and mentioned log1p treatments simultaneous"
        ),
        "treatment_transform": TREATMENT_TRANSFORM,
        "model_treatment_scale": "log(1 + applicants per 1,000 residents)",
        "reported_effect_scale": (
            "sample-average discrete effect of increasing the raw treatment "
            "rate by 1 applicant per 1,000 residents"
        ),
        "discrete_increase_per_1000": DISCRETE_INCREASE_PER_1000,
        "outcome_unit": (
            "topic share on the 0-1 scale; multiply reported effects by 100 "
            "for percentage-point changes"
        ),
        "covariance": "dyad-clustered standard errors",
        "significance_rule": "pretrend p >= 0.05 and DID p < 0.05",
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_outputs(
        output_dir,
        raw_results,
        summary,
        all_changes,
        panels,
        construction_diagnostics,
        sample_diagnostics,
        match_diagnostics,
        coefficient_qc,
        panel_qc,
        settings,
    )
    plots = plot_results(summary, all_changes, output_dir)
    if not plots:
        (output_dir / "PLOT_NOT_CREATED.txt").write_text(
            "Matplotlib is not installed in this Python environment. "
            "All numeric results were created successfully.",
            encoding="utf-8",
        )

    display_columns = [
        "crisis",
        "treatment",
        "topic",
        "origin_main_did_coef",
        "origin_main_did_model_coef",
        "origin_main_did_p",
        "mentioned_main_did_coef",
        "mentioned_main_did_model_coef",
        "mentioned_main_did_p",
        "origin_parallel_trend_pass",
        "mentioned_parallel_trend_pass",
        "origin_final_significant",
        "mentioned_final_significant",
    ]
    print("\nFinished.")
    print(f"Output directory: {output_dir}")
    print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
