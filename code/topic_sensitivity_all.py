"""All sensitivity checks for the latest topic-share dyadic DiD.

One standalone code file containing the latest main-model helpers and both
previously verified sensitivity implementations.  It never edits source data.
Raw CSV estimates are on the 0-1 share scale; the combined focal readout adds
percentage-point columns for manuscript reporting.

Usage: python topic_sensitivity_all.py [--mode all|checks|dyads]
       [--output-dir DIRECTORY]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def _combine_results(root: Path) -> Path:
    sheets = {}
    paths = [
        ("checks", root / "checks" / "topic_robustness_concise_side_summary.csv"),
        ("check_stability", root / "checks" / "topic_robustness_stability_summary.csv"),
        ("dyad_exclusion", root / "dyads" / "topic_sensitivity_scenario_summary.csv"),
        ("leave_one_out", root / "dyads" / "topic_sensitivity_leave_one_country_detail.csv"),
        ("focal_results", root / "dyads" / "topic_sensitivity_main_significant_readout.csv"),
    ]
    for name, path in paths:
        if path.is_file():
            frame = pd.read_csv(path)
            if name == "focal_results":
                for column in frame.columns:
                    if column.endswith(("_coef", "_se", "_ci_low", "_ci_high")):
                        frame[column + "_pp"] = frame[column] * 100
            sheets[name] = frame
    if not sheets:
        raise RuntimeError("No sensitivity results were produced")
    output = root / "topic_sensitivity_combined.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "checks", "dyads"], default="all")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir or Path(__file__).resolve().parent.parent / "results" / (
        "topic_sensitivity_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=True)

    core = _load_embedded("topic_share_did_log1p_aligned")
    bundle_root = Path(__file__).resolve().parent.parent
    core.DEFAULT_TOPIC_FILE = bundle_root / "data" / "cross_country_topic_labeled.xlsx"
    core.DEFAULT_ASYLUM_FILE = bundle_root / "data" / "asylum_data.xlsx"
    core.DEFAULT_OUTPUT_ROOT = bundle_root / "results" / "topic_main"
    core.DEFAULT_OUTPUT_DIR = core.DEFAULT_OUTPUT_ROOT
    core.check_input_files(core.DEFAULT_TOPIC_FILE, core.DEFAULT_ASYLUM_FILE)
    checks = _load_embedded("topic_checks")
    dyads = _load_embedded("topic_dyads")
    checks.DEFAULT_OUTPUT_DIR = root / "checks"
    dyads.DEFAULT_OUTPUT_DIR = root / "dyads"

    # The embedded implementations parse command-line options independently.
    old_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve())]
    try:
        if args.mode in ("all", "checks"):
            checks.main()
        if args.mode in ("all", "dyads"):
            dyads.main()
    finally:
        sys.argv = old_argv

    combined = _combine_results(root)
    settings = {
        "topic_file": str(core.DEFAULT_TOPIC_FILE),
        "asylum_file": str(core.DEFAULT_ASYLUM_FILE),
        "treatment_2015": "first_time_applicants (all first-time applicants)",
        "treatment_2022_non_ukrainian": "non-Ukrainian_first_time_applicant",
        "treatment_2022_ukrainian": "first_time_applicants_ukrainian",
        "model": "PanelOLS; dyad and period fixed effects; origin and mentioned exposures jointly",
        "default_se": "dyad clustered",
        "topic_csv_unit": "proportion (0-1); multiply effect and SE by 100 for percentage points",
        "source_files_edited": False,
        "run_time": datetime.now().isoformat(timespec="seconds"),
    }
    (root / "run_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Combined results:", combined)


if __name__ == "__main__":
    main()

# === BEGIN EMBEDDED topic_share_did_log1p_aligned ===
#@"""
#@Two-sided topic-share DID aligned with the revised sentiment/hate-speech DID.
#@
#@The input is the already topic-labelled tweet-country data.  For every
#@origin-country -> mentioned-country pair and period, the script calculates the
#@share of observations assigned to each topic.  The denominator includes the
#@four substantive topics plus Other.
#@
#@For each two-period comparison, the model is estimated with PanelOLS using
#@dyad fixed effects, period fixed effects, and standard errors clustered by
#@dyad.  The origin-side and mentioned-side log(1 + refugee inflow) interactions
#@enter the same regression.  For interpretation, estimates are converted after
#@estimation into sample-average discrete effects of increasing the original
#@rate by one applicant per 1,000 residents.  This positive rescaling leaves
#@t statistics and two-sided p values unchanged.
#@"""
#@
#@from __future__ import annotations
#@
#@import argparse
#@import math
#@import re
#@from datetime import datetime
#@from pathlib import Path
#@
#@import numpy as np
#@import pandas as pd
#@from linearmodels.panel import PanelOLS
#@try:
#@    import matplotlib.pyplot as plt
#@    from matplotlib.patches import Patch
#@    from matplotlib.ticker import FuncFormatter, MaxNLocator
#@    MATPLOTLIB_AVAILABLE = True
#@except ImportError:
#@    # Robustness estimation imports the data/model helpers but does not draw
#@    # the core script's figures.  Keep those helpers usable in a lean runtime.
#@    plt = None
#@    Patch = None
#@    FuncFormatter = None
#@    MaxNLocator = None
#@    MATPLOTLIB_AVAILABLE = False
#@
#@
#@DEFAULT_TOPIC_FILE = Path(
#@    r"D:\HKU\Europe refugee\Week progress - 2.19\Refugee research materials"
#@    r"\Sub-paper\country level\topic\cross-country_topic_labeled.xlsx"
#@)
#@DEFAULT_ASYLUM_FILE = Path(
#@    r"D:\HKU\Europe refugee\Week progress - 2.19\Refugee research materials"
#@    r"\Sub-paper\country level\data and code avalibility\data availability"
#@    r"\asylum_data.xlsx"
#@)
#@DEFAULT_OUTPUT_ROOT = Path(
#@    r"D:\HKU\Europe refugee\Week progress - 2.19\Refugee research materials"
#@    r"\Sub-paper\country level\DID\topic-DID"
#@)
#@DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "main_results_log1p_aligned"
#@
#@MIN_COUNT_2015 = 20
#@MIN_COUNT_2022 = 30
#@
#@# Treatment and reporting scales.  These match the revised sentiment and
#@# hate-speech DID code.
#@TREATMENT_TRANSFORM = "log1p"
#@REPORT_AVERAGE_DISCRETE_EFFECT = True
#@DISCRETE_INCREASE_PER_1000 = 1.0
#@
#@# Topic shares are estimated on the 0-1 scale.  Multiply only the values sent
#@# to the figures by 100 so that the y axes report percentage-point changes.
#@# Model estimation and the numeric result files remain on their original scale.
#@FIGURE_PERCENTAGE_POINT_SCALE = 100.0
#@
#@PRIMARY_TOPICS = [
#@    "Security",
#@    "Resource Pressure",
#@    "Governance",
#@    "Human Rights",
#@]
#@ALL_TOPICS = PRIMARY_TOPICS + ["Other"]
#@TOPIC_SLUGS = {
#@    "Security": "security",
#@    "Resource Pressure": "resource_pressure",
#@    "Governance": "governance",
#@    "Human Rights": "human_rights",
#@    "Other": "other",
#@}
#@SHARE_COLUMNS = {
#@    topic: f"share_{TOPIC_SLUGS[topic]}" for topic in ALL_TOPICS
#@}
#@
#@
#@# Publication-style figure settings aligned with the main DID code.
#@if MATPLOTLIB_AVAILABLE:
#@    plt.rcParams["font.family"] = "Arial"
#@    plt.rcParams["font.size"] = 10
#@    plt.rcParams["axes.unicode_minus"] = False
#@    plt.rcParams["pdf.fonttype"] = 42
#@    plt.rcParams["ps.fonttype"] = 42
#@
#@TOPIC_COLORS = {
#@    "Governance": "#96C8A3",
#@    "Human Rights": "#8DBEDC",
#@    "Resource Pressure": "#F4BE7E",
#@    "Security": "#E99095",
#@}
#@ZERO_LINE_COLOR = "#A7A7A7"
#@
#@# Match the current sentiment/hate-speech coefficient + dyad-distribution
#@# layout.  The left coefficient panel keeps the original visual weight and a
#@# compact, separately scaled box panel is added on the right.
#@ORIGINAL_MAIN_WIDTH = 5.5
#@BOX_PANEL_WIDTH = 0.58
#@FIG_W = 5.70
#@FIG_H = 5.15
#@MARKER_SIZE = 70
#@MARKER_EDGE_WIDTH = 1.9
#@ERROR_LINE_WIDTH = 2.2
#@BOX_LINE_WIDTH = ERROR_LINE_WIDTH
#@BOX_WIDTH = 0.30
#@SPINE_WIDTH = 1.5
#@TICK_SIZE = 15
#@LABEL_SIZE = 17
#@ANNOT_SIZE = 16
#@SHOW_YLABEL = True
#@SHOW_POST_LABEL = False
#@SAVE_PDF = False
#@
#@
#@def now_tag() -> str:
#@    return datetime.now().strftime("%Y%m%d_%H%M%S")
#@
#@
#@def clean_country_name(value):
#@    if pd.isna(value):
#@        return np.nan
#@    value = re.sub(r"\s+", " ", str(value).strip()).lower()
#@    if value in {"", "nan", "none", "unknown"}:
#@        return np.nan
#@    return value
#@
#@
#@def normalize_topic(value):
#@    if pd.isna(value):
#@        return np.nan
#@    value = re.sub(r"\s+", " ", str(value).strip()).lower()
#@    mapping = {
#@        "security": "Security",
#@        "safety": "Security",
#@        "safety and security": "Security",
#@        "resource pressure": "Resource Pressure",
#@        "resource pressures": "Resource Pressure",
#@        "governance": "Governance",
#@        "human rights": "Human Rights",
#@        "human right": "Human Rights",
#@        "other": "Other",
#@    }
#@    return mapping.get(value, np.nan)
#@
#@
#@def check_input_files(topic_file: Path, asylum_file: Path) -> None:
#@    missing = [str(path) for path in [topic_file, asylum_file] if not path.exists()]
#@    if missing:
#@        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing))
#@
#@
#@# ---------------------------------------------------------------------------
#@# Student-t distribution helpers (self-contained; SciPy is not required)
#@# ---------------------------------------------------------------------------
#@def _beta_continued_fraction(a: float, b: float, x: float) -> float:
#@    max_iterations = 300
#@    epsilon = 3e-14
#@    tiny = 1e-300
#@
#@    qab = a + b
#@    qap = a + 1.0
#@    qam = a - 1.0
#@    c = 1.0
#@    d = 1.0 - qab * x / qap
#@    if abs(d) < tiny:
#@        d = tiny
#@    d = 1.0 / d
#@    h = d
#@
#@    for iteration in range(1, max_iterations + 1):
#@        m2 = 2 * iteration
#@        aa = iteration * (b - iteration) * x / (
#@            (qam + m2) * (a + m2)
#@        )
#@        d = 1.0 + aa * d
#@        if abs(d) < tiny:
#@            d = tiny
#@        c = 1.0 + aa / c
#@        if abs(c) < tiny:
#@            c = tiny
#@        d = 1.0 / d
#@        h *= d * c
#@
#@        aa = -(
#@            (a + iteration)
#@            * (qab + iteration)
#@            * x
#@            / ((a + m2) * (qap + m2))
#@        )
#@        d = 1.0 + aa * d
#@        if abs(d) < tiny:
#@            d = tiny
#@        c = 1.0 + aa / c
#@        if abs(c) < tiny:
#@            c = tiny
#@        d = 1.0 / d
#@        delta = d * c
#@        h *= delta
#@        if abs(delta - 1.0) < epsilon:
#@            break
#@    return h
#@
#@
#@def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
#@    if x <= 0:
#@        return 0.0
#@    if x >= 1:
#@        return 1.0
#@    front = math.exp(
#@        math.lgamma(a + b)
#@        - math.lgamma(a)
#@        - math.lgamma(b)
#@        + a * math.log(x)
#@        + b * math.log1p(-x)
#@    )
#@    if x < (a + 1.0) / (a + b + 2.0):
#@        return front * _beta_continued_fraction(a, b, x) / a
#@    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b
#@
#@
#@def student_t_two_sided_p(t_value: float, df: int) -> float:
#@    if not np.isfinite(t_value) or df <= 0:
#@        return np.nan
#@    x = df / (df + t_value**2)
#@    return regularized_incomplete_beta(df / 2.0, 0.5, x)
#@
#@
#@def student_t_critical_975(df: int) -> float:
#@    if df <= 0:
#@        return np.nan
#@    low, high = 0.0, 20.0
#@    target_p = 0.05
#@    for _ in range(100):
#@        middle = (low + high) / 2.0
#@        if student_t_two_sided_p(middle, df) > target_p:
#@            low = middle
#@        else:
#@            high = middle
#@    return (low + high) / 2.0
#@
#@
#@# ---------------------------------------------------------------------------
#@# Topic-share panel construction
#@# ---------------------------------------------------------------------------
#@def read_topic_labeled_data(topic_file: Path, include_self: bool) -> pd.DataFrame:
#@    required = ["mentioned_country", "origin_country", "post_date", "topic_category"]
#@    frame = pd.read_excel(topic_file, usecols=required).copy()
#@    frame = frame.rename(
#@        columns={
#@            "mentioned_country": "mentioned",
#@            "origin_country": "origin",
#@            "post_date": "date",
#@            "topic_category": "topic",
#@        }
#@    )
#@    frame["origin"] = frame["origin"].map(clean_country_name)
#@    frame["mentioned"] = frame["mentioned"].map(clean_country_name)
#@    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
#@    frame["topic"] = frame["topic"].map(normalize_topic)
#@    frame = frame.dropna(subset=["origin", "mentioned", "date", "topic"]).copy()
#@    frame = frame.loc[frame["date"].dt.year.between(2010, 2024)].copy()
#@    if not include_self:
#@        frame = frame.loc[frame["origin"] != frame["mentioned"]].copy()
#@    frame["year"] = frame["date"].dt.year.astype(int)
#@    frame["pair"] = frame["origin"] + "->" + frame["mentioned"]
#@    return frame[["pair", "origin", "mentioned", "year", "topic"]]
#@
#@
#@def build_topic_share_panel(
#@    topic_data: pd.DataFrame,
#@    year_to_period: dict[int, str],
#@    min_count: int,
#@    comparison_name: str,
#@) -> tuple[pd.DataFrame, dict[str, object]]:
#@    periods = list(dict.fromkeys(year_to_period.values()))
#@    frame = topic_data.loc[topic_data["year"].isin(year_to_period)].copy()
#@    frame["period"] = frame["year"].map(year_to_period)
#@
#@    topic_counts = (
#@        frame.groupby(
#@            ["pair", "origin", "mentioned", "period", "topic"],
#@            as_index=False,
#@        )
#@        .size()
#@        .rename(columns={"size": "topic_count"})
#@    )
#@    pivot = topic_counts.pivot_table(
#@        index=["pair", "origin", "mentioned", "period"],
#@        columns="topic",
#@        values="topic_count",
#@        aggfunc="sum",
#@        fill_value=0,
#@    ).reset_index()
#@    pivot.columns.name = None
#@    for topic in ALL_TOPICS:
#@        if topic not in pivot.columns:
#@            pivot[topic] = 0
#@
#@    pivot["post_count"] = pivot[ALL_TOPICS].sum(axis=1).astype(int)
#@    for topic in ALL_TOPICS:
#@        pivot[SHARE_COLUMNS[topic]] = np.where(
#@            pivot["post_count"] > 0,
#@            pivot[topic] / pivot["post_count"],
#@            np.nan,
#@        )
#@
#@    pairs_before_threshold = int(pivot["pair"].nunique())
#@    rows_before_threshold = int(len(pivot))
#@    pivot = pivot.loc[pivot["post_count"] >= min_count].copy()
#@    pairs_after_threshold = int(pivot["pair"].nunique())
#@    rows_after_threshold = int(len(pivot))
#@
#@    period_counts = pivot.groupby("pair")["period"].nunique()
#@    common_pairs = period_counts.loc[period_counts == len(periods)].index
#@    panel = pivot.loc[pivot["pair"].isin(common_pairs)].copy()
#@
#@    keep_columns = [
#@        "pair",
#@        "origin",
#@        "mentioned",
#@        "period",
#@        "post_count",
#@        *ALL_TOPICS,
#@        *[SHARE_COLUMNS[topic] for topic in ALL_TOPICS],
#@    ]
#@    panel = panel[keep_columns].sort_values(["pair", "period"])
#@    share_sum = panel[[SHARE_COLUMNS[t] for t in ALL_TOPICS]].sum(axis=1)
#@    diagnostics = {
#@        "comparison": comparison_name,
#@        "periods": " | ".join(periods),
#@        "min_count": min_count,
#@        "pairs_before_count_threshold": pairs_before_threshold,
#@        "rows_before_count_threshold": rows_before_threshold,
#@        "pairs_after_count_threshold": pairs_after_threshold,
#@        "rows_after_count_threshold": rows_after_threshold,
#@        "balanced_pairs_before_treatment": int(panel["pair"].nunique()),
#@        "balanced_rows_before_treatment": int(len(panel)),
#@        "max_abs_share_sum_minus_one": (
#@            float(np.max(np.abs(share_sum - 1.0))) if len(panel) else np.nan
#@        ),
#@    }
#@    return panel, diagnostics
#@
#@
#@# ---------------------------------------------------------------------------
#@# Treatment and two-period DID
#@# ---------------------------------------------------------------------------
#@def load_treatment(asylum_file: Path, year: int, treatment: str) -> pd.DataFrame:
#@    frame = pd.read_excel(asylum_file, sheet_name=str(year)).copy()
#@    required = ["host_country", treatment, "population"]
#@    missing = [column for column in required if column not in frame.columns]
#@    if missing:
#@        raise ValueError(f"[asylum {year}] missing columns: {missing}")
#@    frame["country"] = frame["host_country"].map(clean_country_name)
#@    frame[treatment] = pd.to_numeric(frame[treatment], errors="coerce")
#@    frame["population"] = pd.to_numeric(frame["population"], errors="coerce")
#@    frame["raw_treatment_per_1000"] = np.where(
#@        frame["population"] > 0,
#@        1000.0 * frame[treatment] / frame["population"],
#@        np.nan,
#@    )
#@    if (frame["raw_treatment_per_1000"].dropna() < 0).any():
#@        raise ValueError(
#@            f"[asylum {year}] treatment {treatment!r} contains a negative rate"
#@        )
#@
#@    if TREATMENT_TRANSFORM == "log1p":
#@        frame["treatment_model_scale"] = np.log1p(
#@            frame["raw_treatment_per_1000"]
#@        )
#@    elif TREATMENT_TRANSFORM == "raw":
#@        frame["treatment_model_scale"] = frame["raw_treatment_per_1000"]
#@    else:
#@        raise ValueError(
#@            "TREATMENT_TRANSFORM must be either 'log1p' or 'raw'; "
#@            f"received {TREATMENT_TRANSFORM!r}"
#@        )
#@
#@    return frame[
#@        ["country", "raw_treatment_per_1000", "treatment_model_scale"]
#@    ].drop_duplicates("country")
#@
#@
#@def make_change_frame(
#@    panel: pd.DataFrame,
#@    treatment_frame: pd.DataFrame,
#@    baseline_period: str,
#@    comparison_period: str,
#@    crisis: str,
#@    treatment_name: str,
#@    stage: str,
#@) -> tuple[pd.DataFrame, pd.DataFrame]:
#@    value_columns = [SHARE_COLUMNS[topic] for topic in ALL_TOPICS]
#@    baseline = panel.loc[
#@        panel["period"] == baseline_period,
#@        ["pair", "origin", "mentioned", "post_count", *value_columns],
#@    ].copy()
#@    comparison = panel.loc[
#@        panel["period"] == comparison_period,
#@        ["pair", "origin", "mentioned", "post_count", *value_columns],
#@    ].copy()
#@    baseline = baseline.rename(
#@        columns={
#@            "post_count": "baseline_count",
#@            **{column: f"baseline_{column}" for column in value_columns},
#@        }
#@    )
#@    comparison = comparison.rename(
#@        columns={
#@            "post_count": "comparison_count",
#@            **{column: f"comparison_{column}" for column in value_columns},
#@        }
#@    )
#@    changes = baseline.merge(
#@        comparison,
#@        on=["pair", "origin", "mentioned"],
#@        how="inner",
#@        validate="one_to_one",
#@    )
#@
#@    origin_treatment = treatment_frame.rename(
#@        columns={
#@            "country": "origin",
#@            "raw_treatment_per_1000": "raw_treatment_origin_per_1000",
#@            "treatment_model_scale": "treatment_origin_model_scale",
#@        }
#@    )
#@    mentioned_treatment = treatment_frame.rename(
#@        columns={
#@            "country": "mentioned",
#@            "raw_treatment_per_1000": "raw_treatment_mentioned_per_1000",
#@            "treatment_model_scale": "treatment_mentioned_model_scale",
#@        }
#@    )
#@    before_treatment = changes.copy()
#@    changes = changes.merge(origin_treatment, on="origin", how="left")
#@    changes = changes.merge(mentioned_treatment, on="mentioned", how="left")
#@    changes = changes.dropna(
#@        subset=[
#@            "raw_treatment_origin_per_1000",
#@            "raw_treatment_mentioned_per_1000",
#@            "treatment_origin_model_scale",
#@            "treatment_mentioned_model_scale",
#@        ]
#@    ).copy()
#@
#@    for topic in ALL_TOPICS:
#@        share_column = SHARE_COLUMNS[topic]
#@        changes[f"delta_{share_column}"] = (
#@            changes[f"comparison_{share_column}"]
#@            - changes[f"baseline_{share_column}"]
#@        )
#@    changes["crisis"] = crisis
#@    changes["treatment"] = treatment_name
#@    changes["stage"] = stage
#@    changes["baseline_period"] = baseline_period
#@    changes["comparison_period"] = comparison_period
#@
#@    matched_countries = set(treatment_frame.loc[
#@        treatment_frame["treatment_model_scale"].notna(), "country"
#@    ])
#@    unmatched_origin = sorted(set(before_treatment["origin"]) - matched_countries)
#@    unmatched_mentioned = sorted(
#@        set(before_treatment["mentioned"]) - matched_countries
#@    )
#@    match_rows = []
#@    for side, countries in [
#@        ("origin", unmatched_origin),
#@        ("mentioned", unmatched_mentioned),
#@    ]:
#@        for country in countries:
#@            match_rows.append(
#@                {
#@                    "crisis": crisis,
#@                    "treatment": treatment_name,
#@                    "stage": stage,
#@                    "side": side,
#@                    "unmatched_country": country,
#@                }
#@            )
#@    return changes, pd.DataFrame(match_rows)
#@
#@
#@def panel_ols_two_sided(
#@    baseline_outcome: np.ndarray,
#@    comparison_outcome: np.ndarray,
#@    treatment_origin: np.ndarray,
#@    treatment_mentioned: np.ndarray,
#@    pair_ids: np.ndarray,
#@) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
#@    """Estimate the two-sided DID with dyad and period fixed effects."""
#@    baseline_outcome = np.asarray(baseline_outcome, dtype=float)
#@    comparison_outcome = np.asarray(comparison_outcome, dtype=float)
#@    treatment_origin = np.asarray(treatment_origin, dtype=float)
#@    treatment_mentioned = np.asarray(treatment_mentioned, dtype=float)
#@    pair_ids = np.asarray(pair_ids, dtype=object)
#@    valid = (
#@        np.isfinite(baseline_outcome)
#@        & np.isfinite(comparison_outcome)
#@        & np.isfinite(treatment_origin)
#@        & np.isfinite(treatment_mentioned)
#@        & pd.notna(pair_ids)
#@    )
#@    y_baseline = baseline_outcome[valid]
#@    y_comparison = comparison_outcome[valid]
#@    x_origin = treatment_origin[valid]
#@    x_mentioned = treatment_mentioned[valid]
#@    pairs = pair_ids[valid].astype(str)
#@    n = len(pairs)
#@
#@    # After removing dyad and period effects, identification comes from the
#@    # cross-dyad variation in the two treatment intensities.  Check that this
#@    # equivalent first-difference design has full rank before fitting PanelOLS.
#@    change_design = np.column_stack([np.ones(n), x_origin, x_mentioned])
#@    k = change_design.shape[1]
#@
#@    empty_term = {
#@        "coef": np.nan,
#@        "se": np.nan,
#@        "t": np.nan,
#@        "p": np.nan,
#@        "ci_low": np.nan,
#@        "ci_high": np.nan,
#@    }
#@    if n <= k or len(np.unique(pairs)) != n or np.linalg.matrix_rank(change_design) < k:
#@        model = {
#@            "n_pairs": int(n),
#@            "df_resid": int(max(n - k, 0)),
#@            "r_squared": np.nan,
#@            "condition_number": np.nan,
#@        }
#@        return {"origin": empty_term.copy(), "mentioned": empty_term.copy()}, model
#@
#@    baseline = pd.DataFrame(
#@        {
#@            "pair": pairs,
#@            "period_id": 0,
#@            "Y": y_baseline,
#@            "DID_origin": 0.0,
#@            "DID_mentioned": 0.0,
#@        }
#@    )
#@    comparison = pd.DataFrame(
#@        {
#@            "pair": pairs,
#@            "period_id": 1,
#@            "Y": y_comparison,
#@            "DID_origin": x_origin,
#@            "DID_mentioned": x_mentioned,
#@        }
#@    )
#@    panel = (
#@        pd.concat([baseline, comparison], ignore_index=True)
#@        .set_index(["pair", "period_id"])
#@        .sort_index()
#@    )
#@
#@    fitted = PanelOLS(
#@        dependent=panel["Y"],
#@        exog=panel[["DID_origin", "DID_mentioned"]],
#@        entity_effects=True,
#@        time_effects=True,
#@        drop_absorbed=True,
#@    ).fit(
#@        cov_type="clustered",
#@        cluster_entity=True,
#@    )
#@
#@    confidence = fitted.conf_int()
#@
#@    terms = {}
#@    for side, term_name in [
#@        ("origin", "DID_origin"),
#@        ("mentioned", "DID_mentioned"),
#@    ]:
#@        terms[side] = {
#@            "coef": float(fitted.params[term_name]),
#@            "se": float(fitted.std_errors[term_name]),
#@            "t": float(fitted.tstats[term_name]),
#@            "p": float(fitted.pvalues[term_name]),
#@            "ci_low": float(confidence.loc[term_name].iloc[0]),
#@            "ci_high": float(confidence.loc[term_name].iloc[1]),
#@        }
#@
#@    model = {
#@        "n_pairs": int(n),
#@        "df_resid": int(fitted.df_resid),
#@        "r_squared": float(fitted.rsquared),
#@        "condition_number": float(np.linalg.cond(change_design)),
#@    }
#@    return terms, model
#@
#@
#@def average_discrete_factor(raw_treatment: np.ndarray) -> float:
#@    """Mean model-scale change when the raw rate rises by the chosen increment."""
#@    raw_treatment = np.asarray(raw_treatment, dtype=float)
#@    raw_treatment = raw_treatment[np.isfinite(raw_treatment)]
#@    if len(raw_treatment) == 0:
#@        return np.nan
#@    if (raw_treatment < 0).any():
#@        raise ValueError("Raw treatment rates must be non-negative")
#@
#@    increment = float(DISCRETE_INCREASE_PER_1000)
#@    if increment <= 0:
#@        raise ValueError("DISCRETE_INCREASE_PER_1000 must be positive")
#@    if TREATMENT_TRANSFORM == "log1p":
#@        model_scale_changes = (
#@            np.log1p(raw_treatment + increment) - np.log1p(raw_treatment)
#@        )
#@    elif TREATMENT_TRANSFORM == "raw":
#@        model_scale_changes = np.full(len(raw_treatment), increment, dtype=float)
#@    else:
#@        raise ValueError(
#@            "TREATMENT_TRANSFORM must be either 'log1p' or 'raw'; "
#@            f"received {TREATMENT_TRANSFORM!r}"
#@        )
#@    return float(np.mean(model_scale_changes))
#@
#@
#@def convert_to_average_discrete_effect(
#@    term: dict[str, float], raw_treatment: np.ndarray
#@) -> dict[str, float]:
#@    """Keep the fitted log-scale result and add the interpretable AMDE result."""
#@    result = dict(term)
#@    result.update(
#@        {
#@            "model_coef": term["coef"],
#@            "model_se": term["se"],
#@            "model_ci_low": term["ci_low"],
#@            "model_ci_high": term["ci_high"],
#@        }
#@    )
#@    factor = average_discrete_factor(raw_treatment)
#@    result["average_discrete_factor"] = factor
#@
#@    if REPORT_AVERAGE_DISCRETE_EFFECT:
#@        for name in ["coef", "se", "ci_low", "ci_high"]:
#@            result[name] = (
#@                float(term[name]) * factor
#@                if pd.notna(term[name]) and pd.notna(factor)
#@                else np.nan
#@            )
#@    return result
#@
#@
#@def estimate_all_topics(
#@    changes: pd.DataFrame,
#@    crisis: str,
#@    treatment_name: str,
#@    stage: str,
#@    baseline_period: str,
#@    comparison_period: str,
#@) -> pd.DataFrame:
#@    rows = []
#@    for topic in ALL_TOPICS:
#@        share_column = SHARE_COLUMNS[topic]
#@        terms, model = panel_ols_two_sided(
#@            changes[f"baseline_{share_column}"].to_numpy(),
#@            changes[f"comparison_{share_column}"].to_numpy(),
#@            changes["treatment_origin_model_scale"].to_numpy(),
#@            changes["treatment_mentioned_model_scale"].to_numpy(),
#@            changes["pair"].to_numpy(),
#@        )
#@        for side in ["origin", "mentioned"]:
#@            reported_term = convert_to_average_discrete_effect(
#@                terms[side],
#@                changes[f"raw_treatment_{side}_per_1000"].to_numpy(),
#@            )
#@            rows.append(
#@                {
#@                    "crisis": crisis,
#@                    "treatment": treatment_name,
#@                    "stage": stage,
#@                    "topic": topic,
#@                    "primary_topic": topic in PRIMARY_TOPICS,
#@                    "side": side,
#@                    "baseline_period": baseline_period,
#@                    "comparison_period": comparison_period,
#@                    **reported_term,
#@                    **model,
#@                    "n_origins": int(changes["origin"].nunique()),
#@                    "n_mentioned": int(changes["mentioned"].nunique()),
#@                    "outcome_unit": "topic share (0-1)",
#@                    "model_treatment_scale": (
#@                        "log(1 + applicants per 1,000 residents)"
#@                        if TREATMENT_TRANSFORM == "log1p"
#@                        else "applicants per 1,000 residents"
#@                    ),
#@                    "treatment_unit": "applicants per 1,000 residents",
#@                    "reported_effect_scale": (
#@                        "sample-average discrete effect of +1 applicant "
#@                        "per 1,000 residents"
#@                        if REPORT_AVERAGE_DISCRETE_EFFECT
#@                        else "model coefficient"
#@                    ),
#@                    "covariance": "dyad-clustered standard errors",
#@                }
#@            )
#@    return pd.DataFrame(rows)
#@
#@
#@def run_comparison(
#@    panel: pd.DataFrame,
#@    treatment_frame: pd.DataFrame,
#@    baseline_period: str,
#@    comparison_period: str,
#@    crisis: str,
#@    treatment_name: str,
#@    stage: str,
#@) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
#@    changes, match_diagnostics = make_change_frame(
#@        panel,
#@        treatment_frame,
#@        baseline_period,
#@        comparison_period,
#@        crisis,
#@        treatment_name,
#@        stage,
#@    )
#@    results = estimate_all_topics(
#@        changes,
#@        crisis,
#@        treatment_name,
#@        stage,
#@        baseline_period,
#@        comparison_period,
#@    )
#@    sample = {
#@        "crisis": crisis,
#@        "treatment": treatment_name,
#@        "stage": stage,
#@        "baseline_period": baseline_period,
#@        "comparison_period": comparison_period,
#@        "balanced_pairs_before_treatment": int(panel["pair"].nunique()),
#@        "pairs_after_both_treatments_matched": int(changes["pair"].nunique()),
#@        "panel_rows_after_both_treatments_matched": int(2 * len(changes)),
#@        "n_origins": int(changes["origin"].nunique()),
#@        "n_mentioned": int(changes["mentioned"].nunique()),
#@    }
#@    return results, changes, match_diagnostics, sample
#@
#@
#@def build_main_summary(results: pd.DataFrame) -> pd.DataFrame:
#@    rows = []
#@    primary = results.loc[results["primary_topic"]].copy()
#@    for (crisis, treatment, topic), group in primary.groupby(
#@        ["crisis", "treatment", "topic"], sort=False
#@    ):
#@        row = {"crisis": crisis, "treatment": treatment, "topic": topic}
#@        for side in ["origin", "mentioned"]:
#@            pre = group.loc[(group["stage"] == "pretrend") & (group["side"] == side)]
#@            main = group.loc[(group["stage"] == "main_did") & (group["side"] == side)]
#@            pre_row = pre.iloc[0] if not pre.empty else None
#@            main_row = main.iloc[0] if not main.empty else None
#@            for prefix, source in [("pretrend", pre_row), ("main_did", main_row)]:
#@                for statistic in [
#@                    "coef",
#@                    "se",
#@                    "t",
#@                    "p",
#@                    "ci_low",
#@                    "ci_high",
#@                    "model_coef",
#@                    "model_se",
#@                    "model_ci_low",
#@                    "model_ci_high",
#@                    "average_discrete_factor",
#@                    "n_pairs",
#@                    "n_origins",
#@                    "n_mentioned",
#@                ]:
#@                    row[f"{side}_{prefix}_{statistic}"] = (
#@                        source[statistic] if source is not None else np.nan
#@                    )
#@            row[f"{side}_parallel_trend_pass"] = (
#@                bool(pre_row["p"] >= 0.05)
#@                if pre_row is not None and pd.notna(pre_row["p"])
#@                else False
#@            )
#@            row[f"{side}_main_did_significant"] = (
#@                bool(main_row["p"] < 0.05)
#@                if main_row is not None and pd.notna(main_row["p"])
#@                else False
#@            )
#@            row[f"{side}_final_significant"] = bool(
#@                row[f"{side}_parallel_trend_pass"]
#@                and row[f"{side}_main_did_significant"]
#@            )
#@        rows.append(row)
#@    summary = pd.DataFrame(rows)
#@    topic_order = {topic: index for index, topic in enumerate(PRIMARY_TOPICS)}
#@    summary["_topic_order"] = summary["topic"].map(topic_order)
#@    return summary.sort_values(
#@        ["crisis", "treatment", "_topic_order"]
#@    ).drop(columns="_topic_order")
#@
#@
#@def build_compositional_qc(
#@    results: pd.DataFrame, panels: dict[str, pd.DataFrame]
#@) -> pd.DataFrame:
#@    coefficient_qc = (
#@        results.groupby(["crisis", "treatment", "stage", "side"], as_index=False)
#@        .agg(
#@            sum_of_five_topic_coefficients=("coef", "sum"),
#@            sum_of_five_model_coefficients=("model_coef", "sum"),
#@            max_condition_number=("condition_number", "max"),
#@            n_pairs=("n_pairs", "max"),
#@        )
#@    )
#@    panel_errors = []
#@    for comparison, panel in panels.items():
#@        share_sum = panel[[SHARE_COLUMNS[t] for t in ALL_TOPICS]].sum(axis=1)
#@        panel_errors.append(
#@            {
#@                "comparison": comparison,
#@                "max_abs_share_sum_minus_one": float(
#@                    np.max(np.abs(share_sum - 1.0))
#@                ),
#@                "n_panel_rows": int(len(panel)),
#@                "n_pairs": int(panel["pair"].nunique()),
#@            }
#@        )
#@    coefficient_qc["absolute_coefficient_sum"] = coefficient_qc[
#@        "sum_of_five_topic_coefficients"
#@    ].abs()
#@    coefficient_qc["absolute_model_coefficient_sum"] = coefficient_qc[
#@        "sum_of_five_model_coefficients"
#@    ].abs()
#@    return coefficient_qc, pd.DataFrame(panel_errors)
#@
#@
#@def sanitize_filename(value) -> str:
#@    """Keep readable names while replacing characters invalid on Windows."""
#@    text = str(value)
#@    for character in '/\\:*?"<>|':
#@        text = text.replace(character, "_")
#@    return text
#@
#@
#@def format_coef_label(value, ci_low=None, ci_high=None) -> str:
#@    del ci_low, ci_high
#@    if pd.isna(value):
#@        return ""
#@    return f"Effect = {value:.2f} percentage points"
#@
#@
#@def dynamic_formatter(value, position) -> str:
#@    del position
#@    if np.isclose(value, 0, atol=1e-12):
#@        return "0.00"
#@    if abs(value) >= 1:
#@        return f"{value:.1f}"
#@    if abs(value) >= 0.1:
#@        return f"{value:.2f}"
#@    return f"{value:.3f}"
#@
#@
#@def nice_ylim_raw(ci_low: list[float], ci_high: list[float]) -> tuple[float, float]:
#@    values = [value for value in ci_low + ci_high if pd.notna(value)]
#@    if not values:
#@        return -5.0, 5.0
#@    limit = max(abs(min(values)), abs(max(values)), 0.0) * 1.22
#@    limit = max(limit, 3.0)
#@    return -limit, limit
#@
#@
#@def nice_ylim_distribution(values: list[float]) -> tuple[float, float]:
#@    """Return a tight, non-symmetric range for a dyad distribution."""
#@    finite = np.asarray([value for value in values if pd.notna(value)], dtype=float)
#@    if len(finite) == 0:
#@        return -5.0, 5.0
#@    lower = float(np.min(finite))
#@    upper = float(np.max(finite))
#@    span = upper - lower
#@    if span <= 1e-12:
#@        span = max(abs(lower) * 0.20, 1.0)
#@    padding = 0.18 * span
#@    return lower - padding, upper + padding
#@
#@
#@def ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
#@    if pd.isna(ci_low) or pd.isna(ci_high):
#@        return False
#@    return bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
#@
#@
#@def dyad_specific_discrete_effects(
#@    changes: pd.DataFrame,
#@    row: pd.Series,
#@    side: str,
#@    stage: str,
#@) -> pd.DataFrame:
#@    """Return the reported discrete effect for every dyad in one model."""
#@    if side not in {"origin", "mentioned"}:
#@        raise ValueError(f"Unknown side: {side}")
#@    if stage not in {"pretrend", "main_did"}:
#@        raise ValueError(f"Unknown stage: {stage}")
#@
#@    subset = changes.loc[
#@        (changes["crisis"].astype(str) == str(row["crisis"]))
#@        & (changes["treatment"].astype(str) == str(row["treatment"]))
#@        & (changes["stage"].astype(str) == stage)
#@    ].copy()
#@    raw_column = f"raw_treatment_{side}_per_1000"
#@    required = ["pair", "origin", "mentioned", raw_column]
#@    missing = [column for column in required if column not in subset.columns]
#@    if missing:
#@        raise ValueError(f"Dyad-effect input is missing columns: {missing}")
#@
#@    data = subset[required].dropna().copy()
#@    data[raw_column] = pd.to_numeric(data[raw_column], errors="coerce")
#@    data = data.dropna(subset=[raw_column]).copy()
#@    variation = data.groupby("pair")[raw_column].nunique(dropna=False)
#@    if (variation > 1).any():
#@        bad = variation.loc[variation > 1].index.tolist()[:5]
#@        raise ValueError(
#@            f"Treatment varies within dyad for {raw_column}; examples: {bad}"
#@        )
#@    data = data.drop_duplicates("pair").sort_values("pair").reset_index(drop=True)
#@    if (data[raw_column] < 0).any():
#@        raise ValueError(f"Negative treatment rate in {raw_column}")
#@
#@    model_coefficient = float(row[f"{side}_{stage}_model_coef"])
#@    increment = float(DISCRETE_INCREASE_PER_1000)
#@    if TREATMENT_TRANSFORM == "log1p":
#@        contrast = (
#@            np.log1p(data[raw_column] + increment)
#@            - np.log1p(data[raw_column])
#@        )
#@    elif TREATMENT_TRANSFORM == "raw":
#@        contrast = np.full(len(data), increment, dtype=float)
#@    else:
#@        raise ValueError(f"Unknown treatment transform: {TREATMENT_TRANSFORM}")
#@
#@    data = data.rename(columns={raw_column: "raw_treatment_per_1000"})
#@    data["model_scale_contrast"] = contrast
#@    data["dyad_discrete_effect"] = model_coefficient * contrast
#@    data["crisis"] = str(row["crisis"])
#@    data["treatment"] = str(row["treatment"])
#@    data["topic"] = str(row["topic"])
#@    data["side"] = side
#@    data["stage"] = stage
#@
#@    reported = float(row[f"{side}_{stage}_coef"])
#@    dyad_mean = float(data["dyad_discrete_effect"].mean())
#@    if not np.isclose(dyad_mean, reported, rtol=1e-10, atol=1e-12):
#@        raise ValueError(
#@            "Dyad mean does not match the reported average discrete effect: "
#@            f"mean={dyad_mean}, reported={reported}"
#@        )
#@    return data
#@
#@
#@def dyad_effects_for_figure(frame: pd.DataFrame) -> pd.DataFrame:
#@    """Convert dyad-specific topic-share effects to percentage points."""
#@    plotted = frame.copy()
#@    plotted["dyad_discrete_effect"] = (
#@        pd.to_numeric(plotted["dyad_discrete_effect"], errors="coerce")
#@        * FIGURE_PERCENTAGE_POINT_SCALE
#@    )
#@    return plotted
#@
#@
#@def draw_dyad_distribution(axis, x_value: float, frame: pd.DataFrame, color: str):
#@    """Draw Q25-Q75, min-max whiskers, and a black dyad-mean line."""
#@    values = frame["dyad_discrete_effect"].dropna().to_numpy(dtype=float)
#@    if len(values) == 0:
#@        return None
#@    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
#@    minimum = float(np.min(values))
#@    maximum = float(np.max(values))
#@    mean = float(np.mean(values))
#@    axis.bxp(
#@        [{
#@            "label": "",
#@            "whislo": minimum,
#@            "q1": float(q25),
#@            "med": mean,
#@            "q3": float(q75),
#@            "whishi": maximum,
#@            "fliers": [],
#@        }],
#@        positions=[x_value],
#@        widths=BOX_WIDTH,
#@        showfliers=False,
#@        patch_artist=True,
#@        manage_ticks=False,
#@        boxprops={
#@            "facecolor": "none",
#@            "edgecolor": color,
#@            "linewidth": BOX_LINE_WIDTH,
#@        },
#@        medianprops={"color": "none", "linewidth": 0},
#@        whiskerprops={"color": color, "linewidth": BOX_LINE_WIDTH},
#@        capprops={"color": color, "linewidth": BOX_LINE_WIDTH},
#@        zorder=3,
#@    )
#@    mean_half_width = BOX_WIDTH * 0.50
#@    axis.plot(
#@        [x_value - mean_half_width, x_value + mean_half_width],
#@        [mean, mean],
#@        color="#222222",
#@        linewidth=BOX_LINE_WIDTH,
#@        solid_capstyle="butt",
#@        zorder=4,
#@    )
#@    return {
#@        "n": int(len(values)),
#@        "min": minimum,
#@        "q25": float(q25),
#@        "median": float(median),
#@        "mean": mean,
#@        "q75": float(q75),
#@        "max": maximum,
#@    }
#@
#@
#@def style_axis(axis) -> None:
#@    axis.axhline(
#@        0,
#@        linestyle="--",
#@        linewidth=1.0,
#@        color=ZERO_LINE_COLOR,
#@        zorder=1,
#@    )
#@    axis.set_xlim(-0.30, 2.55)
#@    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
#@    axis.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
#@    axis.tick_params(
#@        axis="both",
#@        labelsize=TICK_SIZE,
#@        width=0.8,
#@        length=3.2,
#@        top=False,
#@        right=False,
#@    )
#@    axis.spines["top"].set_visible(False)
#@    axis.spines["right"].set_visible(False)
#@    axis.spines["left"].set_linewidth(SPINE_WIDTH)
#@    axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
#@    axis.spines["left"].set_color("#333333")
#@    axis.spines["bottom"].set_color("#333333")
#@    axis.grid(False)
#@
#@
#@def draw_coef_point(
#@    axis,
#@    x_value: float,
#@    y_value: float,
#@    ci_low: float,
#@    ci_high: float,
#@    color: str,
#@) -> None:
#@    if pd.isna(y_value):
#@        return
#@    if pd.notna(ci_low) and pd.notna(ci_high):
#@        axis.errorbar(
#@            [x_value],
#@            [y_value],
#@            yerr=[[y_value - ci_low], [ci_high - y_value]],
#@            fmt="none",
#@            ecolor=color,
#@            elinewidth=ERROR_LINE_WIDTH,
#@            capsize=0,
#@            alpha=1.0,
#@            zorder=4,
#@        )
#@    axis.scatter(
#@        [x_value],
#@        [y_value],
#@        s=MARKER_SIZE,
#@        facecolors=color if ci_excludes_zero(ci_low, ci_high) else "white",
#@        edgecolors=color,
#@        linewidths=MARKER_EDGE_WIDTH,
#@        alpha=1.0,
#@        zorder=5,
#@    )
#@
#@
#@def period_labels(crisis: str) -> list[str]:
#@    if str(crisis) == "2015":
#@        return [
#@            "2010-2012\n(pre 1)",
#@            "2013-2014\n(baseline)",
#@            "2015-2021\n(post 1)",
#@        ]
#@    return [
#@        "2010-2014\n(pre 1)",
#@        "2015-2021\n(baseline)",
#@        "2022-2024\n(post 1)",
#@    ]
#@
#@
#@def treatment_filename_label(treatment: str) -> str:
#@    if treatment == "non-Ukrainian_first_time_applicant":
#@        return "non-Ukrainian first-time asylum applicant"
#@    if treatment == "first_time_applicants":
#@        return "first-time asylum applicant"
#@    if treatment == "first_time_applicants_ukrainian":
#@        return "first-time asylum applicant (Ukrainians)"
#@    return str(treatment)
#@
#@
#@def figure_filename(row: pd.Series, side: str, extension: str = "png") -> str:
#@    crisis_prefix = "crisis2015" if str(row["crisis"]) == "2015" else "2022"
#@    parts = [
#@        crisis_prefix,
#@        str(row["topic"]).lower(),
#@        treatment_filename_label(str(row["treatment"])),
#@        side,
#@    ]
#@    return sanitize_filename("_".join(parts) + f".{extension}")
#@
#@
#@def draw_single_topic_plot(
#@    row: pd.Series,
#@    side: str,
#@    all_changes: pd.DataFrame,
#@    figure_dir: Path,
#@) -> list[str]:
#@    x_values = [0, 1, 2]
#@    y_values = [
#@        FIGURE_PERCENTAGE_POINT_SCALE
#@        * float(row[f"{side}_pretrend_coef"]),
#@        0.0,
#@        FIGURE_PERCENTAGE_POINT_SCALE
#@        * float(row[f"{side}_main_did_coef"]),
#@    ]
#@    ci_low = [
#@        FIGURE_PERCENTAGE_POINT_SCALE
#@        * float(row[f"{side}_pretrend_ci_low"]),
#@        0.0,
#@        FIGURE_PERCENTAGE_POINT_SCALE
#@        * float(row[f"{side}_main_did_ci_low"]),
#@    ]
#@    ci_high = [
#@        FIGURE_PERCENTAGE_POINT_SCALE
#@        * float(row[f"{side}_pretrend_ci_high"]),
#@        0.0,
#@        FIGURE_PERCENTAGE_POINT_SCALE
#@        * float(row[f"{side}_main_did_ci_high"]),
#@    ]
#@    color = TOPIC_COLORS[str(row["topic"])]
#@    post_dyads = dyad_effects_for_figure(
#@        dyad_specific_discrete_effects(
#@            all_changes,
#@            row,
#@            side,
#@            "main_did",
#@        )
#@    )
#@
#@    figure, (axis, box_axis) = plt.subplots(
#@        1,
#@        2,
#@        figsize=(FIG_W, FIG_H),
#@        gridspec_kw={
#@            "width_ratios": [ORIGINAL_MAIN_WIDTH, BOX_PANEL_WIDTH],
#@            "wspace": 0.085,
#@        },
#@    )
#@    # The middle period is the omitted reference category, not an estimated
#@    # zero-effect coefficient; label it on the x axis but do not draw a point.
#@    for index in (0, 2):
#@        draw_coef_point(
#@            axis,
#@            x_values[index],
#@            y_values[index],
#@            ci_low[index],
#@            ci_high[index],
#@            color,
#@        )
#@
#@    axis.set_ylim(*nice_ylim_raw(ci_low, ci_high))
#@    style_axis(axis)
#@    axis.set_xticks(x_values)
#@    axis.set_xticklabels(period_labels(str(row["crisis"])))
#@    axis.set_xlabel("")
#@    if SHOW_YLABEL:
#@        axis.set_ylabel(
#@            "Average change in topic share\n(percentage points)",
#@            fontsize=LABEL_SIZE,
#@            labelpad=9,
#@        )
#@    if SHOW_POST_LABEL and pd.notna(y_values[2]):
#@        axis.annotate(
#@            format_coef_label(y_values[2], ci_low[2], ci_high[2]),
#@            xy=(x_values[2], y_values[2]),
#@            xytext=(9, 0),
#@            textcoords="offset points",
#@            ha="left",
#@            va="center",
#@            fontsize=ANNOT_SIZE,
#@            color=color,
#@            fontweight="bold",
#@            zorder=6,
#@        )
#@
#@    # The right panel follows the current sentiment figure: Q25-Q75 box,
#@    # min-max whiskers, and a black mean line, all on a separate tight scale.
#@    draw_dyad_distribution(box_axis, 0.0, post_dyads, color)
#@    box_values = post_dyads["dyad_discrete_effect"].dropna().tolist()
#@    box_ymin, box_ymax = nice_ylim_distribution(box_values)
#@    box_axis.set_ylim(box_ymin, box_ymax)
#@    box_axis.set_xlim(-0.46, 0.46)
#@    box_axis.set_xticks([0.0])
#@    post_period = (
#@        "2015–2021" if str(row["crisis"]) == "2015" else "2022–2024"
#@    )
#@    box_axis.set_xticklabels([post_period])
#@    box_axis.set_xlabel("")
#@    box_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
#@    box_axis.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
#@    box_axis.yaxis.tick_right()
#@    box_axis.yaxis.set_label_position("right")
#@    box_axis.tick_params(
#@        axis="both",
#@        labelsize=max(TICK_SIZE - 1, 8),
#@        width=0.8,
#@        length=3.0,
#@        top=False,
#@        left=False,
#@        right=True,
#@    )
#@    box_axis.spines["top"].set_visible(False)
#@    box_axis.spines["left"].set_visible(False)
#@    box_axis.spines["right"].set_linewidth(SPINE_WIDTH)
#@    box_axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
#@    box_axis.spines["right"].set_color("#333333")
#@    box_axis.spines["bottom"].set_color("#333333")
#@    box_axis.set_facecolor("white")
#@    box_axis.set_title("")
#@    box_axis.set_ylabel(
#@        "Dyad estimates\n(percentage points)",
#@        fontsize=max(LABEL_SIZE - 1, 9),
#@        labelpad=8,
#@    )
#@    if box_ymin <= 0 <= box_ymax:
#@        box_axis.axhline(
#@            0,
#@            linestyle="--",
#@            linewidth=0.9,
#@            color=ZERO_LINE_COLOR,
#@            zorder=1,
#@        )
#@
#@    figure.tight_layout()
#@    png_path = figure_dir / figure_filename(row, side, "png")
#@    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
#@    created = [str(png_path)]
#@    if SAVE_PDF:
#@        pdf_path = figure_dir / figure_filename(row, side, "pdf")
#@        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
#@        created.append(str(pdf_path))
#@    plt.close(figure)
#@    return created
#@
#@
#@def draw_combined_2022_total_figure(
#@    summary: pd.DataFrame,
#@    all_changes: pd.DataFrame,
#@    figure_dir: Path,
#@    reference_png: Path | None = None,
#@) -> Path:
#@    """Combine the two significant 2022 total-inflow mentioned-side results."""
#@    specifications = [
#@        ("Governance", -0.055, -0.18),
#@        ("Human Rights", 0.055, 0.18),
#@    ]
#@    selected = []
#@    for topic, coefficient_offset, box_position in specifications:
#@        rows = summary.loc[
#@            (summary["crisis"].astype(str) == "2022")
#@            & (summary["treatment"].astype(str) == "non-Ukrainian_first_time_applicant")
#@            & (summary["topic"].astype(str) == topic)
#@        ]
#@        if len(rows) != 1:
#@            raise RuntimeError(
#@                f"Expected one 2022 non-Ukrainian_first_time_applicant row for {topic}; found {len(rows)}"
#@            )
#@        row = rows.iloc[0]
#@        if not bool(row["mentioned_final_significant"]):
#@            raise RuntimeError(
#@                f"The requested combined result is not final-significant: {topic}"
#@            )
#@        dyads = dyad_specific_discrete_effects(
#@            all_changes,
#@            row,
#@            "mentioned",
#@            "main_did",
#@        )
#@        dyads = dyad_effects_for_figure(dyads)
#@        selected.append((topic, row, coefficient_offset, box_position, dyads))
#@
#@    figure, (axis, box_axis) = plt.subplots(
#@        1,
#@        2,
#@        figsize=(FIG_W, FIG_H),
#@        gridspec_kw={
#@            "width_ratios": [ORIGINAL_MAIN_WIDTH, BOX_PANEL_WIDTH],
#@            "wspace": 0.085,
#@        },
#@    )
#@    x_values = np.array([0.0, 1.0, 2.0])
#@    coefficient_scale_values = []
#@    box_scale_values = []
#@
#@    for topic, row, coefficient_offset, box_position, dyads in selected:
#@        color = TOPIC_COLORS[topic]
#@        values = [
#@            FIGURE_PERCENTAGE_POINT_SCALE
#@            * float(row["mentioned_pretrend_coef"]),
#@            0.0,
#@            FIGURE_PERCENTAGE_POINT_SCALE
#@            * float(row["mentioned_main_did_coef"]),
#@        ]
#@        lows = [
#@            FIGURE_PERCENTAGE_POINT_SCALE
#@            * float(row["mentioned_pretrend_ci_low"]),
#@            0.0,
#@            FIGURE_PERCENTAGE_POINT_SCALE
#@            * float(row["mentioned_main_did_ci_low"]),
#@        ]
#@        highs = [
#@            FIGURE_PERCENTAGE_POINT_SCALE
#@            * float(row["mentioned_pretrend_ci_high"]),
#@            0.0,
#@            FIGURE_PERCENTAGE_POINT_SCALE
#@            * float(row["mentioned_main_did_ci_high"]),
#@        ]
#@        for index in (0, 2):
#@            draw_coef_point(
#@                axis,
#@                float(x_values[index] + coefficient_offset),
#@                values[index],
#@                lows[index],
#@                highs[index],
#@                color,
#@            )
#@        coefficient_scale_values.extend(values + lows + highs)
#@        draw_dyad_distribution(box_axis, box_position, dyads, color)
#@        box_scale_values.extend(
#@            dyads["dyad_discrete_effect"].dropna().astype(float).tolist()
#@        )
#@
#@    coefficient_scale_values = [
#@        value for value in coefficient_scale_values if pd.notna(value)
#@    ]
#@    axis.set_ylim(
#@        *nice_ylim_raw(coefficient_scale_values, coefficient_scale_values)
#@    )
#@    style_axis(axis)
#@    axis.set_xticks(x_values)
#@    axis.set_xticklabels(period_labels("2022"))
#@    axis.set_xlabel("")
#@    axis.set_ylabel(
#@        "Average change in topic share\n(percentage points)",
#@        fontsize=LABEL_SIZE,
#@        labelpad=9,
#@    )
#@
#@    box_ymin, box_ymax = nice_ylim_distribution(box_scale_values)
#@    box_axis.set_ylim(box_ymin, box_ymax)
#@    box_axis.set_xlim(-0.46, 0.46)
#@    box_axis.set_xticks([0.0])
#@    box_axis.set_xticklabels(["2022–2024"])
#@    box_axis.set_xlabel("")
#@    box_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
#@    box_axis.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))
#@    box_axis.yaxis.tick_right()
#@    box_axis.yaxis.set_label_position("right")
#@    box_axis.tick_params(
#@        axis="both",
#@        labelsize=max(TICK_SIZE - 1, 8),
#@        width=0.8,
#@        length=3.0,
#@        top=False,
#@        left=False,
#@        right=True,
#@    )
#@    box_axis.spines["top"].set_visible(False)
#@    box_axis.spines["left"].set_visible(False)
#@    box_axis.spines["right"].set_linewidth(SPINE_WIDTH)
#@    box_axis.spines["bottom"].set_linewidth(SPINE_WIDTH)
#@    box_axis.spines["right"].set_color("#333333")
#@    box_axis.spines["bottom"].set_color("#333333")
#@    box_axis.set_ylabel(
#@        "Dyad estimates\n(percentage points)",
#@        fontsize=max(LABEL_SIZE - 1, 9),
#@        labelpad=8,
#@    )
#@    box_axis.set_title("")
#@    if box_ymin <= 0 <= box_ymax:
#@        box_axis.axhline(
#@            0,
#@            linestyle="--",
#@            linewidth=0.9,
#@            color=ZERO_LINE_COLOR,
#@            zorder=1,
#@        )
#@
#@    figure.tight_layout()
#@    output_path = (
#@        figure_dir
#@        / "2022_total_first_time_asylum_seekers_two_significant_topics.png"
#@    )
#@    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
#@    plt.close(figure)
#@
#@    # bbox_inches="tight" can make the exported PNG dimensions differ slightly
#@    # even when figsize is identical.  Match this one extra combined figure to
#@    # the actual pixel dimensions of a regular topic figure, without changing
#@    # any model, axis, colour, or annotation settings.
#@    if reference_png is not None and reference_png.exists():
#@        from PIL import Image
#@
#@        with Image.open(reference_png) as reference_image:
#@            target_width, target_height = reference_image.size
#@
#@        with Image.open(output_path) as source_image:
#@            source_image = source_image.convert("RGB")
#@            source_width, source_height = source_image.size
#@
#@            if (source_width, source_height) != (target_width, target_height):
#@                scale = min(
#@                    target_width / source_width,
#@                    target_height / source_height,
#@                )
#@                resized_width = max(1, round(source_width * scale))
#@                resized_height = max(1, round(source_height * scale))
#@                resampling = getattr(Image, "Resampling", Image).LANCZOS
#@                resized = source_image.resize(
#@                    (resized_width, resized_height),
#@                    resampling,
#@                )
#@                canvas = Image.new(
#@                    "RGB",
#@                    (target_width, target_height),
#@                    "white",
#@                )
#@                left = (target_width - resized_width) // 2
#@                top = (target_height - resized_height) // 2
#@                canvas.paste(resized, (left, top))
#@                canvas.save(output_path, dpi=(600, 600))
#@
#@    return output_path
#@
#@
#@def draw_topic_color_legend(figure_dir: Path) -> Path:
#@    """Save the four-topic colour key used across all topic figures."""
#@    figure, axis = plt.subplots(figsize=(10.2, 0.72))
#@    axis.axis("off")
#@    handles = [
#@        Patch(facecolor=TOPIC_COLORS[topic], edgecolor="none", label=topic)
#@        for topic in ["Governance", "Human Rights", "Resource Pressure", "Security"]
#@    ]
#@    axis.legend(
#@        handles=handles,
#@        loc="center",
#@        ncol=4,
#@        frameon=False,
#@        fontsize=15,
#@        columnspacing=1.35,
#@        handlelength=2.1,
#@        handletextpad=0.28,
#@    )
#@    output_path = figure_dir / "topic_color_legend.png"
#@    figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
#@    plt.close(figure)
#@    return output_path
#@
#@
#@def plot_results(
#@    summary: pd.DataFrame,
#@    all_changes: pd.DataFrame,
#@    output_dir: Path,
#@) -> list[str]:
#@    """Draw sentiment-style coefficient and dyad-distribution topic figures."""
#@    figure_dir = output_dir / "figures"
#@    figure_dir.mkdir(parents=True, exist_ok=True)
#@    created = []
#@    for _, row in summary.iterrows():
#@        for side in ["origin", "mentioned"]:
#@            created.extend(
#@                draw_single_topic_plot(row, side, all_changes, figure_dir)
#@            )
#@    reference_png = next(
#@        (Path(path) for path in created if Path(path).suffix.lower() == ".png"),
#@        None,
#@    )
#@    created.append(
#@        str(
#@            draw_combined_2022_total_figure(
#@                summary,
#@                all_changes,
#@                figure_dir,
#@                reference_png=reference_png,
#@            )
#@        )
#@    )
#@    created.append(str(draw_topic_color_legend(figure_dir)))
#@
#@    # Export the exact dyad values represented by the post-period box plots.
#@    dyad_parts = []
#@    for _, row in summary.iterrows():
#@        for side in ["origin", "mentioned"]:
#@            dyad_parts.append(
#@                dyad_specific_discrete_effects(
#@                    all_changes,
#@                    row,
#@                    side,
#@                    "main_did",
#@                )
#@            )
#@    pd.concat(dyad_parts, ignore_index=True).to_csv(
#@        output_dir / "topic_share_post_dyad_specific_effects.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    return created
#@
#@
#@def save_outputs(
#@    output_dir: Path,
#@    raw_results: pd.DataFrame,
#@    summary: pd.DataFrame,
#@    changes: pd.DataFrame,
#@    panels: dict[str, pd.DataFrame],
#@    construction_diagnostics: list[dict[str, object]],
#@    sample_diagnostics: list[dict[str, object]],
#@    match_diagnostics: pd.DataFrame,
#@    coefficient_qc: pd.DataFrame,
#@    panel_qc: pd.DataFrame,
#@    settings: dict[str, object],
#@) -> None:
#@    output_dir.mkdir(parents=True, exist_ok=True)
#@    raw_results.to_csv(
#@        output_dir / "topic_share_all_results.csv", index=False, encoding="utf-8-sig"
#@    )
#@    summary.to_csv(
#@        output_dir / "topic_share_main_summary.csv", index=False, encoding="utf-8-sig"
#@    )
#@    changes.to_csv(
#@        output_dir / "topic_share_pair_changes.csv", index=False, encoding="utf-8-sig"
#@    )
#@    for name, panel in panels.items():
#@        panel.to_csv(
#@            output_dir / f"topic_share_panel_{name}.csv",
#@            index=False,
#@            encoding="utf-8-sig",
#@        )
#@    pd.DataFrame(construction_diagnostics).to_csv(
#@        output_dir / "topic_share_panel_construction.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    pd.DataFrame(sample_diagnostics).to_csv(
#@        output_dir / "topic_share_sample_counts.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    match_diagnostics.to_csv(
#@        output_dir / "topic_share_unmatched_countries.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    coefficient_qc.to_csv(
#@        output_dir / "topic_share_compositional_coefficient_qc.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    panel_qc.to_csv(
#@        output_dir / "topic_share_compositional_panel_qc.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    pd.DataFrame([settings]).to_csv(
#@        output_dir / "topic_share_run_settings.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@
#@    warnings = []
#@    for sample in sample_diagnostics:
#@        n_pairs = sample["pairs_after_both_treatments_matched"]
#@        if n_pairs < 30:
#@            warnings.append(
#@                f"{sample['crisis']} / {sample['treatment']} / {sample['stage']}: "
#@                f"only {n_pairs} balanced country pairs after treatment matching."
#@            )
#@    failed = summary[
#@        (~summary["origin_parallel_trend_pass"])
#@        | (~summary["mentioned_parallel_trend_pass"])
#@    ]
#@    if not failed.empty:
#@        warnings.append(
#@            "At least one pre-trend test fails for the specifications "
#@            "listed in topic_share_main_summary.csv."
#@        )
#@    (output_dir / "ANALYSIS_WARNINGS.txt").write_text(
#@        "\n".join(warnings) if warnings else "No automatic warnings.",
#@        encoding="utf-8",
#@    )
#@
#@
#@def parse_args() -> argparse.Namespace:
#@    parser = argparse.ArgumentParser(description=__doc__)
#@    parser.add_argument("--topic-file", type=Path, default=DEFAULT_TOPIC_FILE)
#@    parser.add_argument("--asylum-file", type=Path, default=DEFAULT_ASYLUM_FILE)
#@    parser.add_argument("--output-dir", type=Path, default=None)
#@    parser.add_argument("--min-count-2015", type=int, default=MIN_COUNT_2015)
#@    parser.add_argument("--min-count-2022", type=int, default=MIN_COUNT_2022)
#@    parser.add_argument("--include-self", action="store_true")
#@    # Jupyter/IPython automatically appends arguments such as
#@    # ``-f .../kernel-<id>.json``.  They are unrelated to this analysis and
#@    # would make parse_args() terminate the notebook with SystemExit: 2.
#@    # parse_known_args() keeps all of this script's real command-line options
#@    # while safely ignoring those notebook-kernel arguments.
#@    args, _notebook_arguments = parser.parse_known_args()
#@    return args
#@
#@
#@def main() -> None:
#@    args = parse_args()
#@    output_dir = (
#@        args.output_dir
#@        if args.output_dir is not None
#@        else DEFAULT_OUTPUT_DIR
#@    )
#@    check_input_files(args.topic_file, args.asylum_file)
#@
#@    print("Reading topic-labelled data...")
#@    topic_data = read_topic_labeled_data(args.topic_file, args.include_self)
#@    print(
#@        f"Usable labelled rows: {len(topic_data):,}; "
#@        f"origins={topic_data['origin'].nunique()}; "
#@        f"mentioned={topic_data['mentioned'].nunique()}"
#@    )
#@
#@    panel_2015_pre, construction_2015_pre = build_topic_share_panel(
#@        topic_data,
#@        {
#@            2010: "2010-2012",
#@            2011: "2010-2012",
#@            2012: "2010-2012",
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@        },
#@        args.min_count_2015,
#@        "2015_pretrend",
#@    )
#@    panel_2015_main, construction_2015_main = build_topic_share_panel(
#@        topic_data,
#@        {
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@        },
#@        args.min_count_2015,
#@        "2015_main",
#@    )
#@    panel_2022_pre, construction_2022_pre = build_topic_share_panel(
#@        topic_data,
#@        {
#@            **{year: "2010-2014" for year in range(2010, 2015)},
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@        },
#@        args.min_count_2022,
#@        "2022_pretrend",
#@    )
#@    panel_2022_main, construction_2022_main = build_topic_share_panel(
#@        topic_data,
#@        {
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@            **{year: "2022-2024" for year in range(2022, 2025)},
#@        },
#@        args.min_count_2022,
#@        "2022_main",
#@    )
#@    panels = {
#@        "2015_pretrend": panel_2015_pre,
#@        "2015_main": panel_2015_main,
#@        "2022_pretrend": panel_2022_pre,
#@        "2022_main": panel_2022_main,
#@    }
#@    construction_diagnostics = [
#@        construction_2015_pre,
#@        construction_2015_main,
#@        construction_2022_pre,
#@        construction_2022_main,
#@    ]
#@
#@    comparison_specs = []
#@    treatment_2015 = load_treatment(args.asylum_file, 2015, "first_time_applicants")
#@    comparison_specs.extend(
#@        [
#@            (
#@                panel_2015_pre,
#@                treatment_2015,
#@                "2013-2014",
#@                "2010-2012",
#@                "2015",
#@                "first_time_applicants",
#@                "pretrend",
#@            ),
#@            (
#@                panel_2015_main,
#@                treatment_2015,
#@                "2013-2014",
#@                "2015-2021",
#@                "2015",
#@                "first_time_applicants",
#@                "main_did",
#@            ),
#@        ]
#@    )
#@    for treatment_name in ["non-Ukrainian_first_time_applicant", "first_time_applicants_ukrainian"]:
#@        treatment_2022 = load_treatment(
#@            args.asylum_file, 2022, treatment_name
#@        )
#@        comparison_specs.extend(
#@            [
#@                (
#@                    panel_2022_pre,
#@                    treatment_2022,
#@                    "2015-2021",
#@                    "2010-2014",
#@                    "2022",
#@                    treatment_name,
#@                    "pretrend",
#@                ),
#@                (
#@                    panel_2022_main,
#@                    treatment_2022,
#@                    "2015-2021",
#@                    "2022-2024",
#@                    "2022",
#@                    treatment_name,
#@                    "main_did",
#@                ),
#@            ]
#@        )
#@
#@    result_parts = []
#@    change_parts = []
#@    match_parts = []
#@    sample_diagnostics = []
#@    for spec in comparison_specs:
#@        result, changes, match, sample = run_comparison(*spec)
#@        result_parts.append(result)
#@        change_parts.append(changes)
#@        if not match.empty:
#@            match_parts.append(match)
#@        sample_diagnostics.append(sample)
#@
#@    raw_results = pd.concat(result_parts, ignore_index=True)
#@    summary = build_main_summary(raw_results)
#@    all_changes = pd.concat(change_parts, ignore_index=True)
#@    match_diagnostics = (
#@        pd.concat(match_parts, ignore_index=True)
#@        if match_parts
#@        else pd.DataFrame(
#@            columns=["crisis", "treatment", "stage", "side", "unmatched_country"]
#@        )
#@    )
#@    coefficient_qc, panel_qc = build_compositional_qc(raw_results, panels)
#@
#@    settings = {
#@        "topic_file": str(args.topic_file),
#@        "asylum_file": str(args.asylum_file),
#@        "include_self": args.include_self,
#@        "min_count_2015": args.min_count_2015,
#@        "min_count_2022": args.min_count_2022,
#@        "primary_topics": " | ".join(PRIMARY_TOPICS),
#@        "denominator_topics": " | ".join(ALL_TOPICS),
#@        "model": (
#@            "PanelOLS with dyad and period fixed effects; "
#@            "origin and mentioned log1p treatments simultaneous"
#@        ),
#@        "treatment_transform": TREATMENT_TRANSFORM,
#@        "model_treatment_scale": "log(1 + applicants per 1,000 residents)",
#@        "reported_effect_scale": (
#@            "sample-average discrete effect of increasing the raw treatment "
#@            "rate by 1 applicant per 1,000 residents"
#@        ),
#@        "discrete_increase_per_1000": DISCRETE_INCREASE_PER_1000,
#@        "outcome_unit": (
#@            "topic share on the 0-1 scale; multiply reported effects by 100 "
#@            "for percentage-point changes"
#@        ),
#@        "covariance": "dyad-clustered standard errors",
#@        "significance_rule": "pretrend p >= 0.05 and DID p < 0.05",
#@        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
#@    }
#@    save_outputs(
#@        output_dir,
#@        raw_results,
#@        summary,
#@        all_changes,
#@        panels,
#@        construction_diagnostics,
#@        sample_diagnostics,
#@        match_diagnostics,
#@        coefficient_qc,
#@        panel_qc,
#@        settings,
#@    )
#@    plots = plot_results(summary, all_changes, output_dir)
#@    if not plots:
#@        (output_dir / "PLOT_NOT_CREATED.txt").write_text(
#@            "Matplotlib is not installed in this Python environment. "
#@            "All numeric results were created successfully.",
#@            encoding="utf-8",
#@        )
#@
#@    display_columns = [
#@        "crisis",
#@        "treatment",
#@        "topic",
#@        "origin_main_did_coef",
#@        "origin_main_did_model_coef",
#@        "origin_main_did_p",
#@        "mentioned_main_did_coef",
#@        "mentioned_main_did_model_coef",
#@        "mentioned_main_did_p",
#@        "origin_parallel_trend_pass",
#@        "mentioned_parallel_trend_pass",
#@        "origin_final_significant",
#@        "mentioned_final_significant",
#@    ]
#@    print("\nFinished.")
#@    print(f"Output directory: {output_dir}")
#@    print(summary[display_columns].to_string(index=False))
#@
#@
#@if __name__ == "__main__":
#@    main()
# === END EMBEDDED topic_share_did_log1p_aligned ===


# === BEGIN EMBEDDED topic_checks ===
#@"""
#@Robustness checks for the two-sided topic-share DID.
#@
#@This script updates the five checks in the manuscript's original DID
#@robustness script so that every baseline check follows the revised main
#@topic-share DID:
#@
#@1. observation-count weighted DID using log1p exposure;
#@2. pair, two-way origin/mentioned, origin, and mentioned clustering using
#@   log1p exposure;
#@3. raw-level treatment intensity as the alternative functional form;
#@4. pre-crisis placebo DID using log1p exposure;
#@5. the count threshold increased by 20 observations using log1p exposure.
#@
#@The outcome is the pair-period share of each topic.  The denominator contains
#@Security, Resource Pressure, Governance, Human Rights, and Other.  Origin- and
#@mentioned-country treatment intensities enter the same regression.  Every
#@check is estimated with PanelOLS using dyad and period fixed effects.  The
#@default covariance estimator clusters standard errors by dyad; the clustering
#@checks replace this with the stated origin and/or mentioned-country clusters.
#@"""
#@
#@from __future__ import annotations
#@
#@import argparse
#@import json
#@import sys
#@from datetime import datetime
#@from pathlib import Path
#@
#@import numpy as np
#@import pandas as pd
#@from linearmodels.panel import PanelOLS
#@
#@
#@# ``__file__`` exists when this file is run as a normal Python script, but it
#@# is not defined when the code is pasted/executed directly in Jupyter.  In a
#@# notebook, first look in the current working directory; otherwise use the
#@# project's fixed topic-DID directory requested for these final files.
#@NOTEBOOK_PROJECT_DIR = Path(
#@    r"D:\HKU\Europe refugee\Week progress - 2.19\Refugee research materials"
#@    r"\Sub-paper\country level\DID\topic-DID"
#@)
#@if "__file__" in globals():
#@    SCRIPT_DIR = Path(__file__).resolve().parent
#@elif (Path.cwd() / "topic_share_did.py").exists():
#@    SCRIPT_DIR = Path.cwd()
#@else:
#@    SCRIPT_DIR = NOTEBOOK_PROJECT_DIR
#@
#@if str(SCRIPT_DIR) not in sys.path:
#@    sys.path.insert(0, str(SCRIPT_DIR))
#@
#@try:
#@    import topic_share_did_log1p_aligned as core  # noqa: E402
#@except ImportError:
#@    # Permit the revised core to retain its older filename, but never allow an
#@    # old raw-treatment core to be used silently.
#@    import topic_share_did as core  # type: ignore  # noqa: E402
#@
#@if getattr(core, "TREATMENT_TRANSFORM", None) != "log1p":
#@    raise RuntimeError(
#@        "The robustness script requires the revised log1p topic-share DID "
#@        "core. Place topic_share_did_log1p_aligned.py beside this script."
#@    )
#@if not getattr(core, "REPORT_AVERAGE_DISCRETE_EFFECT", False):
#@    raise RuntimeError(
#@        "The revised core must report sample-average discrete effects."
#@    )
#@if not hasattr(core, "panel_ols_two_sided"):
#@    raise RuntimeError(
#@        "The robustness script requires the PanelOLS version of the topic-share "
#@        "DID core. Update topic_share_did_log1p_aligned.py first."
#@    )
#@
#@
#@DEFAULT_OUTPUT_DIR = core.DEFAULT_OUTPUT_ROOT / "robustness_results_log1p_aligned"
#@CHECK_ORDER = [
#@    "weighted",
#@    "cluster_pair",
#@    "cluster_origin_mentioned",
#@    "cluster_origin",
#@    "cluster_mentioned",
#@    "raw_treatment",
#@    "placebo",
#@    "threshold_plus20",
#@]
#@
#@STANDARD_PERIODS = {
#@    "2015_pretrend": {
#@        "mapping": {
#@            2010: "2010-2012",
#@            2011: "2010-2012",
#@            2012: "2010-2012",
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@        },
#@        "baseline": "2013-2014",
#@        "comparison": "2010-2012",
#@        "stage": "pretrend",
#@    },
#@    "2015_main": {
#@        "mapping": {
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@        },
#@        "baseline": "2013-2014",
#@        "comparison": "2015-2021",
#@        "stage": "main_did",
#@    },
#@    "2022_pretrend": {
#@        "mapping": {
#@            **{year: "2010-2014" for year in range(2010, 2015)},
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@        },
#@        "baseline": "2015-2021",
#@        "comparison": "2010-2014",
#@        "stage": "pretrend",
#@    },
#@    "2022_main": {
#@        "mapping": {
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@            **{year: "2022-2024" for year in range(2022, 2025)},
#@        },
#@        "baseline": "2015-2021",
#@        "comparison": "2022-2024",
#@        "stage": "main_did",
#@    },
#@}
#@
#@PLACEBO_2022_PERIODS = {
#@    "2022_placebo_pretrend": {
#@        "mapping": {
#@            2010: "2010-2012",
#@            2011: "2010-2012",
#@            2012: "2010-2012",
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@        },
#@        "baseline": "2013-2014",
#@        "comparison": "2010-2012",
#@        "stage": "pretrend",
#@    },
#@    "2022_placebo_main": {
#@        "mapping": {
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@            **{year: "2015-2021" for year in range(2015, 2022)},
#@        },
#@        "baseline": "2013-2014",
#@        "comparison": "2015-2021",
#@        "stage": "placebo_did",
#@    },
#@}
#@
#@PLACEBO_2015_SPECS = [
#@    {
#@        "name": "v1_standard_wider",
#@        "pre_mapping": {
#@            2010: "2010-2011",
#@            2011: "2010-2011",
#@            2012: "2012-2013",
#@            2013: "2012-2013",
#@        },
#@        "pre_baseline": "2012-2013",
#@        "pre_comparison": "2010-2011",
#@        "post_mapping": {
#@            2012: "2012-2013",
#@            2013: "2012-2013",
#@            2014: "2014",
#@        },
#@        "post_baseline": "2012-2013",
#@        "post_comparison": "2014",
#@    },
#@    {
#@        "name": "v2_widest",
#@        "pre_mapping": {
#@            2010: "2010-2012",
#@            2011: "2010-2012",
#@            2012: "2010-2012",
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@        },
#@        "pre_baseline": "2013-2014",
#@        "pre_comparison": "2010-2012",
#@        "post_mapping": {
#@            2011: "2011-2012",
#@            2012: "2011-2012",
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@        },
#@        "post_baseline": "2011-2012",
#@        "post_comparison": "2013-2014",
#@    },
#@    {
#@        "name": "v3_shifted",
#@        "pre_mapping": {
#@            2010: "2010-2012",
#@            2011: "2010-2012",
#@            2012: "2010-2012",
#@            2013: "2013",
#@        },
#@        "pre_baseline": "2013",
#@        "pre_comparison": "2010-2012",
#@        "post_mapping": {
#@            2011: "2011-2012",
#@            2012: "2011-2012",
#@            2013: "2013-2014",
#@            2014: "2013-2014",
#@        },
#@        "post_baseline": "2011-2012",
#@        "post_comparison": "2013-2014",
#@    },
#@]
#@
#@MODEL_CASES = [
#@    ("2015", "first_time_applicants", 2015),
#@    ("2022", "non-Ukrainian_first_time_applicant", 2022),
#@    ("2022", "first_time_applicants_ukrainian", 2022),
#@]
#@
#@
#@def parse_args() -> argparse.Namespace:
#@    parser = argparse.ArgumentParser(description=__doc__)
#@    parser.add_argument("--topic-file", type=Path, default=core.DEFAULT_TOPIC_FILE)
#@    parser.add_argument("--asylum-file", type=Path, default=core.DEFAULT_ASYLUM_FILE)
#@    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
#@    parser.add_argument("--min-count-2015", type=int, default=core.MIN_COUNT_2015)
#@    parser.add_argument("--min-count-2022", type=int, default=core.MIN_COUNT_2022)
#@    parser.add_argument("--include-self", action="store_true")
#@    # Ignore Jupyter/IPython's automatic ``-f kernel-<id>.json`` arguments.
#@    # All options defined above are still parsed normally when the file is run
#@    # from a terminal or through a notebook's %run command.
#@    args, _notebook_arguments = parser.parse_known_args()
#@    return args
#@
#@
#@def make_change_frame(
#@    panel: pd.DataFrame,
#@    treatment_frame: pd.DataFrame,
#@    baseline_period: str,
#@    comparison_period: str,
#@    crisis: str,
#@    treatment_name: str,
#@    stage: str,
#@) -> tuple[pd.DataFrame, pd.DataFrame]:
#@    """Create the balanced two-period dyad data used by PanelOLS."""
#@    changes, unmatched = core.make_change_frame(
#@        panel,
#@        treatment_frame,
#@        baseline_period,
#@        comparison_period,
#@        crisis,
#@        treatment_name,
#@        stage,
#@    )
#@    # Profiling the two weighted pair observations gives this exact WLS weight
#@    # for their first difference: w0*w1/(w0+w1).
#@    denominator = changes["baseline_count"] + changes["comparison_count"]
#@    changes["first_difference_weight"] = np.where(
#@        denominator > 0,
#@        changes["baseline_count"] * changes["comparison_count"] / denominator,
#@        np.nan,
#@    )
#@    return changes, unmatched
#@
#@
#@def estimate_model(
#@    changes: pd.DataFrame,
#@    topic: str,
#@    cluster_mode: str,
#@    use_weights: bool,
#@    treatment_scale: str,
#@) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
#@    share_column = core.SHARE_COLUMNS[topic]
#@    baseline_outcome_column = f"baseline_{share_column}"
#@    comparison_outcome_column = f"comparison_{share_column}"
#@    if treatment_scale == "log1p":
#@        origin_treatment_column = "treatment_origin_model_scale"
#@        mentioned_treatment_column = "treatment_mentioned_model_scale"
#@    elif treatment_scale == "level":
#@        origin_treatment_column = "raw_treatment_origin_per_1000"
#@        mentioned_treatment_column = "raw_treatment_mentioned_per_1000"
#@    else:
#@        raise ValueError(f"Unknown treatment scale: {treatment_scale}")
#@
#@    needed = [
#@        baseline_outcome_column,
#@        comparison_outcome_column,
#@        "baseline_count",
#@        "comparison_count",
#@        origin_treatment_column,
#@        mentioned_treatment_column,
#@        "raw_treatment_origin_per_1000",
#@        "raw_treatment_mentioned_per_1000",
#@        "origin",
#@        "mentioned",
#@        "pair",
#@    ]
#@    frame = changes.dropna(subset=needed).copy()
#@    change_design = np.column_stack(
#@        [
#@            np.ones(len(frame)),
#@            frame[origin_treatment_column].astype(float).to_numpy(),
#@            frame[mentioned_treatment_column].astype(float).to_numpy(),
#@        ]
#@    )
#@    n, k = change_design.shape
#@    model = {
#@        "n_pairs": int(n),
#@        "df_resid": int(max(n - k, 0)),
#@        "n_origin_clusters": int(frame["origin"].nunique()),
#@        "n_mentioned_clusters": int(frame["mentioned"].nunique()),
#@        "r_squared": np.nan,
#@        "condition_number": np.nan,
#@        "model_status": "failed",
#@        "model_error": "",
#@    }
#@    empty = {
#@        "coef": np.nan,
#@        "se": np.nan,
#@        "t": np.nan,
#@        "p": np.nan,
#@        "ci_low": np.nan,
#@        "ci_high": np.nan,
#@        "term_status": "failed",
#@    }
#@    if n <= k:
#@        model["model_error"] = f"Insufficient pair rows: n={n}, k={k}."
#@        return {"origin": empty.copy(), "mentioned": empty.copy()}, model
#@    if frame["pair"].astype(str).nunique() != n:
#@        model["model_error"] = "Each retained dyad must occur exactly once in changes."
#@        return {"origin": empty.copy(), "mentioned": empty.copy()}, model
#@    if np.linalg.matrix_rank(change_design) < k:
#@        model["model_error"] = "Treatment design matrix is rank deficient."
#@        return {"origin": empty.copy(), "mentioned": empty.copy()}, model
#@
#@    if use_weights:
#@        baseline_weights = frame["baseline_count"].astype(float).to_numpy()
#@        comparison_weights = frame["comparison_count"].astype(float).to_numpy()
#@        if (
#@            np.any(~np.isfinite(baseline_weights))
#@            or np.any(~np.isfinite(comparison_weights))
#@            or np.any(baseline_weights <= 0)
#@            or np.any(comparison_weights <= 0)
#@        ):
#@            model["model_error"] = "Non-positive or invalid period observation weights."
#@            return {"origin": empty.copy(), "mentioned": empty.copy()}, model
#@    else:
#@        baseline_weights = np.ones(n, dtype=float)
#@        comparison_weights = np.ones(n, dtype=float)
#@
#@    pairs = frame["pair"].astype(str).to_numpy()
#@    origins = frame["origin"].astype(str).to_numpy()
#@    mentioned = frame["mentioned"].astype(str).to_numpy()
#@    origin_exposure = frame[origin_treatment_column].astype(float).to_numpy()
#@    mentioned_exposure = frame[mentioned_treatment_column].astype(float).to_numpy()
#@
#@    baseline = pd.DataFrame(
#@        {
#@            "pair": pairs,
#@            "period_id": 0,
#@            "origin": origins,
#@            "mentioned": mentioned,
#@            "Y": frame[baseline_outcome_column].astype(float).to_numpy(),
#@            "DID_origin": 0.0,
#@            "DID_mentioned": 0.0,
#@            "model_weight": baseline_weights,
#@        }
#@    )
#@    comparison = pd.DataFrame(
#@        {
#@            "pair": pairs,
#@            "period_id": 1,
#@            "origin": origins,
#@            "mentioned": mentioned,
#@            "Y": frame[comparison_outcome_column].astype(float).to_numpy(),
#@            "DID_origin": origin_exposure,
#@            "DID_mentioned": mentioned_exposure,
#@            "model_weight": comparison_weights,
#@        }
#@    )
#@    panel = (
#@        pd.concat([baseline, comparison], ignore_index=True)
#@        .set_index(["pair", "period_id"])
#@        .sort_index()
#@    )
#@
#@    model_kwargs = {
#@        "dependent": panel["Y"],
#@        "exog": panel[["DID_origin", "DID_mentioned"]],
#@        "entity_effects": True,
#@        "time_effects": True,
#@        "drop_absorbed": True,
#@    }
#@    if use_weights:
#@        model_kwargs["weights"] = panel["model_weight"]
#@
#@    try:
#@        estimator = PanelOLS(**model_kwargs)
#@        if cluster_mode == "pair":
#@            fitted = estimator.fit(
#@                cov_type="clustered",
#@                cluster_entity=True,
#@            )
#@        elif cluster_mode == "origin":
#@            clusters = pd.DataFrame(
#@                {"origin": pd.Categorical(panel["origin"]).codes},
#@                index=panel.index,
#@            )
#@            fitted = estimator.fit(cov_type="clustered", clusters=clusters)
#@        elif cluster_mode == "mentioned":
#@            clusters = pd.DataFrame(
#@                {"mentioned": pd.Categorical(panel["mentioned"]).codes},
#@                index=panel.index,
#@            )
#@            fitted = estimator.fit(cov_type="clustered", clusters=clusters)
#@        elif cluster_mode == "origin_mentioned":
#@            clusters = pd.DataFrame(
#@                {
#@                    "origin": pd.Categorical(panel["origin"]).codes,
#@                    "mentioned": pd.Categorical(panel["mentioned"]).codes,
#@                },
#@                index=panel.index,
#@            )
#@            fitted = estimator.fit(cov_type="clustered", clusters=clusters)
#@        else:
#@            raise ValueError(f"Unknown cluster mode: {cluster_mode}")
#@    except Exception as exc:
#@        model["model_error"] = str(exc)
#@        return {"origin": empty.copy(), "mentioned": empty.copy()}, model
#@
#@    confidence = fitted.conf_int()
#@    profiled_weights = frame["first_difference_weight"].astype(float).to_numpy()
#@    condition_design = change_design
#@    if use_weights and np.all(np.isfinite(profiled_weights)):
#@        condition_design = change_design * np.sqrt(profiled_weights)[:, None]
#@    model.update(
#@        {
#@            "df_resid": int(fitted.df_resid),
#@            "r_squared": float(fitted.rsquared),
#@            "condition_number": float(np.linalg.cond(condition_design)),
#@            "model_status": "success",
#@        }
#@    )
#@    terms: dict[str, dict[str, object]] = {}
#@    invalid_terms = []
#@    for side, term_name in [
#@        ("origin", "DID_origin"),
#@        ("mentioned", "DID_mentioned"),
#@    ]:
#@        coefficient = float(fitted.params[term_name])
#@        standard_error = float(fitted.std_errors[term_name])
#@        if not np.isfinite(standard_error) or standard_error <= 0:
#@            terms[side] = {
#@                **empty,
#@                "coef": coefficient,
#@                "term_status": "invalid_covariance",
#@            }
#@            invalid_terms.append(side)
#@            continue
#@        terms[side] = {
#@            "coef": coefficient,
#@            "se": standard_error,
#@            "t": float(fitted.tstats[term_name]),
#@            "p": float(fitted.pvalues[term_name]),
#@            "ci_low": float(confidence.loc[term_name].iloc[0]),
#@            "ci_high": float(confidence.loc[term_name].iloc[1]),
#@            "term_status": "success",
#@        }
#@    if invalid_terms:
#@        model["model_status"] = "partial_invalid_covariance"
#@        model["model_error"] = (
#@            "Non-positive cluster variance for: " + ", ".join(invalid_terms)
#@        )
#@    return terms, model
#@
#@
#@def estimate_all_topics(
#@    changes: pd.DataFrame,
#@    check_name: str,
#@    check_family: str,
#@    cluster_mode: str,
#@    use_weights: bool,
#@    treatment_scale: str,
#@    crisis: str,
#@    treatment_name: str,
#@    stage: str,
#@    baseline_period: str,
#@    comparison_period: str,
#@    min_count: int,
#@    placebo_spec: str = "",
#@) -> pd.DataFrame:
#@    rows = []
#@    for topic in core.ALL_TOPICS:
#@        terms, model = estimate_model(
#@            changes, topic, cluster_mode, use_weights, treatment_scale
#@        )
#@        for side in ["origin", "mentioned"]:
#@            model_term = dict(terms[side])
#@            reported_term = dict(model_term)
#@            reported_term.update(
#@                {
#@                    "model_coef": model_term["coef"],
#@                    "model_se": model_term["se"],
#@                    "model_ci_low": model_term["ci_low"],
#@                    "model_ci_high": model_term["ci_high"],
#@                }
#@            )
#@            if treatment_scale == "log1p":
#@                reported_term = core.convert_to_average_discrete_effect(
#@                    model_term,
#@                    changes[
#@                        f"raw_treatment_{side}_per_1000"
#@                    ].to_numpy(),
#@                )
#@            else:
#@                reported_term["average_discrete_factor"] = 1.0
#@            rows.append(
#@                {
#@                    "check_name": check_name,
#@                    "check_family": check_family,
#@                    "cluster_mode": cluster_mode,
#@                    "use_weights": use_weights,
#@                    "treatment_scale": treatment_scale,
#@                    "min_count": min_count,
#@                    "placebo_spec": placebo_spec,
#@                    "crisis": crisis,
#@                    "treatment": treatment_name,
#@                    "stage": stage,
#@                    "topic": topic,
#@                    "primary_topic": topic in core.PRIMARY_TOPICS,
#@                    "side": side,
#@                    "baseline_period": baseline_period,
#@                    "comparison_period": comparison_period,
#@                    **reported_term,
#@                    **model,
#@                    "outcome_unit": "topic share (0-1)",
#@                    "model_treatment_scale": (
#@                        "log(1 + applicants per 1,000 residents)"
#@                        if treatment_scale == "log1p"
#@                        else "applicants per 1,000 residents"
#@                    ),
#@                    "reported_effect_scale": (
#@                        "sample-average discrete effect of +1 applicant "
#@                        "per 1,000 residents"
#@                    ),
#@                }
#@            )
#@    return pd.DataFrame(rows)
#@
#@
#@def load_treatment_variant(
#@    asylum_file: Path,
#@    year: int,
#@    treatment_name: str,
#@    treatment_scale: str,
#@) -> pd.DataFrame:
#@    # The revised core supplies both the raw rate and log1p model scale.  The
#@    # requested scale is selected later in estimate_model(), so no source data
#@    # or treatment frame is overwritten here.
#@    if treatment_scale not in {"log1p", "level"}:
#@        raise ValueError(f"Unknown treatment scale: {treatment_scale}")
#@    return core.load_treatment(asylum_file, year, treatment_name)
#@
#@
#@def build_panel_cached(
#@    cache: dict[tuple[object, ...], tuple[pd.DataFrame, dict[str, object]]],
#@    topic_data: pd.DataFrame,
#@    mapping: dict[int, str],
#@    min_count: int,
#@    name: str,
#@) -> tuple[pd.DataFrame, dict[str, object]]:
#@    key = (tuple(sorted(mapping.items())), min_count, name)
#@    if key not in cache:
#@        cache[key] = core.build_topic_share_panel(
#@            topic_data, mapping, min_count, name
#@        )
#@    return cache[key]
#@
#@
#@def run_one_comparison(
#@    topic_data: pd.DataFrame,
#@    panel_cache: dict[tuple[object, ...], tuple[pd.DataFrame, dict[str, object]]],
#@    treatment_cache: dict[tuple[object, ...], pd.DataFrame],
#@    asylum_file: Path,
#@    check_name: str,
#@    check_family: str,
#@    cluster_mode: str,
#@    use_weights: bool,
#@    treatment_scale: str,
#@    min_count: int,
#@    crisis: str,
#@    treatment_name: str,
#@    treatment_year: int,
#@    period_name: str,
#@    period_spec: dict[str, object],
#@    placebo_spec: str = "",
#@) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, dict[str, object]]:
#@    panel, construction = build_panel_cached(
#@        panel_cache,
#@        topic_data,
#@        period_spec["mapping"],
#@        min_count,
#@        period_name,
#@    )
#@    treatment_key = (treatment_year, treatment_name, treatment_scale)
#@    if treatment_key not in treatment_cache:
#@        treatment_cache[treatment_key] = load_treatment_variant(
#@            asylum_file, treatment_year, treatment_name, treatment_scale
#@        )
#@    changes, unmatched = make_change_frame(
#@        panel,
#@        treatment_cache[treatment_key],
#@        str(period_spec["baseline"]),
#@        str(period_spec["comparison"]),
#@        crisis,
#@        treatment_name,
#@        str(period_spec["stage"]),
#@    )
#@    result = estimate_all_topics(
#@        changes=changes,
#@        check_name=check_name,
#@        check_family=check_family,
#@        cluster_mode=cluster_mode,
#@        use_weights=use_weights,
#@        treatment_scale=treatment_scale,
#@        crisis=crisis,
#@        treatment_name=treatment_name,
#@        stage=str(period_spec["stage"]),
#@        baseline_period=str(period_spec["baseline"]),
#@        comparison_period=str(period_spec["comparison"]),
#@        min_count=min_count,
#@        placebo_spec=placebo_spec,
#@    )
#@    sample = {
#@        "check_name": check_name,
#@        "crisis": crisis,
#@        "treatment": treatment_name,
#@        "stage": str(period_spec["stage"]),
#@        "baseline_period": str(period_spec["baseline"]),
#@        "comparison_period": str(period_spec["comparison"]),
#@        "min_count": min_count,
#@        "placebo_spec": placebo_spec,
#@        "balanced_pairs_before_treatment": int(panel["pair"].nunique()),
#@        "pairs_after_both_treatments_matched": int(changes["pair"].nunique()),
#@        "n_origins": int(changes["origin"].nunique()),
#@        "n_mentioned": int(changes["mentioned"].nunique()),
#@        "median_first_difference_weight": (
#@            float(changes["first_difference_weight"].median())
#@            if len(changes)
#@            else np.nan
#@        ),
#@    }
#@    construction_row = {
#@        "check_name": check_name,
#@        "crisis": crisis,
#@        "treatment": treatment_name,
#@        "stage": str(period_spec["stage"]),
#@        "placebo_spec": placebo_spec,
#@        **construction,
#@    }
#@    return result, sample, unmatched, construction_row
#@
#@
#@def check_specs(min_count_2015: int, min_count_2022: int) -> list[dict[str, object]]:
#@    return [
#@        {
#@            "check_name": "weighted",
#@            "check_family": "weighted_did",
#@            "cluster_mode": "pair",
#@            "use_weights": True,
#@            "treatment_scale": "log1p",
#@            "min_2015": min_count_2015,
#@            "min_2022": min_count_2022,
#@        },
#@        *[
#@            {
#@                "check_name": f"cluster_{mode}",
#@                "check_family": "cluster_robustness",
#@                "cluster_mode": mode,
#@                "use_weights": False,
#@                "treatment_scale": "log1p",
#@                "min_2015": min_count_2015,
#@                "min_2022": min_count_2022,
#@            }
#@            for mode in ["pair", "origin_mentioned", "origin", "mentioned"]
#@        ],
#@        {
#@            "check_name": "raw_treatment",
#@            "check_family": "functional_form_raw_level",
#@            "cluster_mode": "pair",
#@            "use_weights": False,
#@            "treatment_scale": "level",
#@            "min_2015": min_count_2015,
#@            "min_2022": min_count_2022,
#@        },
#@        {
#@            "check_name": "threshold_plus20",
#@            "check_family": "threshold_plus20",
#@            "cluster_mode": "pair",
#@            "use_weights": False,
#@            "treatment_scale": "log1p",
#@            "min_2015": min_count_2015 + 20,
#@            "min_2022": min_count_2022 + 20,
#@        },
#@    ]
#@
#@
#@def model_is_estimable(result: pd.DataFrame) -> bool:
#@    primary = result.loc[result["primary_topic"]]
#@    return (
#@        not primary.empty
#@        and primary["coef"].notna().all()
#@        and (primary["model_status"] != "failed").all()
#@    )
#@
#@
#@def run_standard_checks(
#@    topic_data: pd.DataFrame,
#@    asylum_file: Path,
#@    min_count_2015: int,
#@    min_count_2022: int,
#@) -> tuple[list[pd.DataFrame], list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
#@    results = []
#@    samples = []
#@    unmatched = []
#@    constructions = []
#@    panel_cache: dict[tuple[object, ...], tuple[pd.DataFrame, dict[str, object]]] = {}
#@    treatment_cache: dict[tuple[object, ...], pd.DataFrame] = {}
#@    for check in check_specs(min_count_2015, min_count_2022):
#@        print(f"Running {check['check_name']}...")
#@        for crisis, treatment_name, treatment_year in MODEL_CASES:
#@            min_count = int(check["min_2015"] if crisis == "2015" else check["min_2022"])
#@            for suffix in ["pretrend", "main"]:
#@                period_key = f"{crisis}_{suffix}"
#@                result, sample, match, construction = run_one_comparison(
#@                    topic_data=topic_data,
#@                    panel_cache=panel_cache,
#@                    treatment_cache=treatment_cache,
#@                    asylum_file=asylum_file,
#@                    check_name=str(check["check_name"]),
#@                    check_family=str(check["check_family"]),
#@                    cluster_mode=str(check["cluster_mode"]),
#@                    use_weights=bool(check["use_weights"]),
#@                    treatment_scale=str(check["treatment_scale"]),
#@                    min_count=min_count,
#@                    crisis=crisis,
#@                    treatment_name=treatment_name,
#@                    treatment_year=treatment_year,
#@                    period_name=period_key,
#@                    period_spec=STANDARD_PERIODS[period_key],
#@                )
#@                results.append(result)
#@                samples.append(sample)
#@                if not match.empty:
#@                    match = match.copy()
#@                    match["check_name"] = check["check_name"]
#@                    unmatched.append(match)
#@                constructions.append(construction)
#@    return results, samples, unmatched, constructions
#@
#@
#@def run_placebo_checks(
#@    topic_data: pd.DataFrame,
#@    asylum_file: Path,
#@    min_count_2015: int,
#@    min_count_2022: int,
#@) -> tuple[list[pd.DataFrame], list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
#@    results = []
#@    samples = []
#@    unmatched = []
#@    constructions = []
#@    attempts = []
#@    panel_cache: dict[tuple[object, ...], tuple[pd.DataFrame, dict[str, object]]] = {}
#@    treatment_cache: dict[tuple[object, ...], pd.DataFrame] = {}
#@
#@    print("Running placebo...")
#@    for treatment_name in ["non-Ukrainian_first_time_applicant", "first_time_applicants_ukrainian"]:
#@        for period_name, period_spec in PLACEBO_2022_PERIODS.items():
#@            result, sample, match, construction = run_one_comparison(
#@                topic_data=topic_data,
#@                panel_cache=panel_cache,
#@                treatment_cache=treatment_cache,
#@                asylum_file=asylum_file,
#@                check_name="placebo",
#@                check_family="placebo_did",
#@                cluster_mode="pair",
#@                use_weights=False,
#@                treatment_scale="log1p",
#@                min_count=min_count_2022,
#@                crisis="2022",
#@                treatment_name=treatment_name,
#@                treatment_year=2022,
#@                period_name=period_name,
#@                period_spec=period_spec,
#@                placebo_spec="2022_standard",
#@            )
#@            results.append(result)
#@            samples.append(sample)
#@            if not match.empty:
#@                match = match.copy()
#@                match["check_name"] = "placebo"
#@                unmatched.append(match)
#@            constructions.append(construction)
#@
#@    chosen = None
#@    for spec in PLACEBO_2015_SPECS:
#@        trial_results = []
#@        trial_samples = []
#@        trial_unmatched = []
#@        trial_constructions = []
#@        for stage, mapping_key, baseline_key, comparison_key in [
#@            ("pretrend", "pre_mapping", "pre_baseline", "pre_comparison"),
#@            ("placebo_did", "post_mapping", "post_baseline", "post_comparison"),
#@        ]:
#@            period_spec = {
#@                "mapping": spec[mapping_key],
#@                "baseline": spec[baseline_key],
#@                "comparison": spec[comparison_key],
#@                "stage": stage,
#@            }
#@            result, sample, match, construction = run_one_comparison(
#@                topic_data=topic_data,
#@                panel_cache=panel_cache,
#@                treatment_cache=treatment_cache,
#@                asylum_file=asylum_file,
#@                check_name="placebo",
#@                check_family="placebo_did",
#@                cluster_mode="pair",
#@                use_weights=False,
#@                treatment_scale="log1p",
#@                min_count=min_count_2015,
#@                crisis="2015",
#@                treatment_name="first_time_applicants",
#@                treatment_year=2015,
#@                period_name=f"2015_placebo_{spec['name']}_{stage}",
#@                period_spec=period_spec,
#@                placebo_spec=str(spec["name"]),
#@            )
#@            trial_results.append(result)
#@            trial_samples.append(sample)
#@            if not match.empty:
#@                match = match.copy()
#@                match["check_name"] = "placebo"
#@                trial_unmatched.append(match)
#@            trial_constructions.append(construction)
#@        estimable = all(model_is_estimable(part) for part in trial_results)
#@        attempts.append(
#@            {
#@                "crisis": "2015",
#@                "placebo_spec": spec["name"],
#@                "estimable": estimable,
#@                "pretrend_pairs": trial_samples[0]["pairs_after_both_treatments_matched"],
#@                "placebo_pairs": trial_samples[1]["pairs_after_both_treatments_matched"],
#@                "selected": False,
#@            }
#@        )
#@        if estimable:
#@            chosen = spec["name"]
#@            results.extend(trial_results)
#@            samples.extend(trial_samples)
#@            unmatched.extend(trial_unmatched)
#@            constructions.extend(trial_constructions)
#@            attempts[-1]["selected"] = True
#@            break
#@    if chosen is None:
#@        # Preserve the last attempted model and its explicit failure diagnostics.
#@        results.extend(trial_results)
#@        samples.extend(trial_samples)
#@        unmatched.extend(trial_unmatched)
#@        constructions.extend(trial_constructions)
#@    return results, samples, unmatched, constructions, attempts
#@
#@
#@def build_concise_summary(results: pd.DataFrame) -> pd.DataFrame:
#@    rows = []
#@    primary = results.loc[results["primary_topic"]].copy()
#@    for keys, group in primary.groupby(
#@        [
#@            "check_name",
#@            "check_family",
#@            "cluster_mode",
#@            "use_weights",
#@            "treatment_scale",
#@            "min_count",
#@            "placebo_spec",
#@            "crisis",
#@            "treatment",
#@            "topic",
#@            "side",
#@        ],
#@        dropna=False,
#@        sort=False,
#@    ):
#@        (
#@            check_name,
#@            check_family,
#@            cluster_mode,
#@            use_weights,
#@            treatment_scale,
#@            min_count,
#@            placebo_spec,
#@            crisis,
#@            treatment,
#@            topic,
#@            side,
#@        ) = keys
#@        pre = group.loc[group["stage"] == "pretrend"]
#@        effect_stage = "placebo_did" if check_name == "placebo" else "main_did"
#@        effect = group.loc[group["stage"] == effect_stage]
#@        pre_row = pre.iloc[0] if len(pre) else None
#@        effect_row = effect.iloc[0] if len(effect) else None
#@        row = {
#@            "check_name": check_name,
#@            "check_family": check_family,
#@            "cluster_mode": cluster_mode,
#@            "use_weights": use_weights,
#@            "treatment_scale": treatment_scale,
#@            "min_count": min_count,
#@            "placebo_spec": placebo_spec,
#@            "crisis": crisis,
#@            "treatment": treatment,
#@            "topic": topic,
#@            "side": side,
#@            "effect_stage": effect_stage,
#@        }
#@        for prefix, source in [("pretrend", pre_row), ("effect", effect_row)]:
#@            for column in [
#@                "coef",
#@                "se",
#@                "t",
#@                "p",
#@                "ci_low",
#@                "ci_high",
#@                "n_pairs",
#@                "n_origin_clusters",
#@                "n_mentioned_clusters",
#@                "model_status",
#@                "model_error",
#@            ]:
#@                row[f"{prefix}_{column}"] = (
#@                    source[column] if source is not None else np.nan
#@                )
#@        pre_pass = (
#@            pre_row is not None
#@            and pd.notna(pre_row["p"])
#@            and float(pre_row["p"]) >= 0.05
#@        )
#@        effect_significant = (
#@            effect_row is not None
#@            and pd.notna(effect_row["p"])
#@            and float(effect_row["p"]) < 0.05
#@        )
#@        row["parallel_trend_pass"] = bool(pre_pass)
#@        row["effect_significant"] = bool(effect_significant)
#@        row["final_model_significant"] = bool(
#@            pre_pass and effect_significant
#@        )
#@        if check_name == "placebo":
#@            # Retain the original robustness script's model-level rule: the
#@            # placebo is a false positive only when its own pretrend passes and
#@            # its placebo DID is significant.  A failed placebo pretrend is
#@            # inconclusive, not evidence that the placebo check passed.
#@            row["placebo_false_positive"] = bool(
#@                pre_pass and effect_significant
#@            )
#@            row["robustness_pass"] = bool(
#@                pre_pass and not effect_significant
#@            )
#@            row["final_substantive_significant"] = False
#@        else:
#@            row["placebo_false_positive"] = False
#@            row["robustness_pass"] = bool(pre_pass and effect_significant)
#@            row["final_substantive_significant"] = bool(
#@                pre_pass and effect_significant
#@            )
#@        rows.append(row)
#@    summary = pd.DataFrame(rows)
#@    check_rank = {name: index for index, name in enumerate(CHECK_ORDER)}
#@    topic_rank = {name: index for index, name in enumerate(core.PRIMARY_TOPICS)}
#@    summary["_check_rank"] = summary["check_name"].map(check_rank)
#@    summary["_topic_rank"] = summary["topic"].map(topic_rank)
#@    return summary.sort_values(
#@        ["_check_rank", "crisis", "treatment", "_topic_rank", "side"]
#@    ).drop(columns=["_check_rank", "_topic_rank"])
#@
#@
#@def build_stability_summary(concise: pd.DataFrame) -> pd.DataFrame:
#@    substantive = concise.loc[concise["check_name"] != "placebo"].copy()
#@    baseline = substantive.loc[substantive["check_name"] == "cluster_pair", [
#@        "crisis", "treatment", "topic", "side", "effect_coef"
#@    ]].rename(columns={"effect_coef": "baseline_pair_coef"})
#@    substantive = substantive.merge(
#@        baseline,
#@        on=["crisis", "treatment", "topic", "side"],
#@        how="left",
#@    )
#@    substantive["same_sign_as_pair"] = (
#@        np.sign(substantive["effect_coef"])
#@        == np.sign(substantive["baseline_pair_coef"])
#@    )
#@    rows = []
#@    for keys, group in substantive.groupby(
#@        ["crisis", "treatment", "topic", "side"], sort=False
#@    ):
#@        crisis, treatment, topic, side = keys
#@        valid = group["effect_coef"].notna()
#@        row = {
#@            "crisis": crisis,
#@            "treatment": treatment,
#@            "topic": topic,
#@            "side": side,
#@            "baseline_pair_coef": group["baseline_pair_coef"].dropna().iloc[0]
#@            if group["baseline_pair_coef"].notna().any()
#@            else np.nan,
#@            "n_substantive_specs": int(len(group)),
#@            "n_estimable_effects": int(valid.sum()),
#@            "n_same_sign_as_pair": int(group.loc[valid, "same_sign_as_pair"].sum()),
#@            "n_parallel_trend_pass": int(group["parallel_trend_pass"].sum()),
#@            "n_final_significant": int(
#@                group["final_substantive_significant"].sum()
#@            ),
#@            "min_effect_coef": group["effect_coef"].min(),
#@            "max_effect_coef": group["effect_coef"].max(),
#@        }
#@        rows.append(row)
#@    stability = pd.DataFrame(rows)
#@    placebo = concise.loc[concise["check_name"] == "placebo", [
#@        "crisis",
#@        "treatment",
#@        "topic",
#@        "side",
#@        "effect_coef",
#@        "effect_p",
#@        "placebo_false_positive",
#@        "placebo_spec",
#@    ]].rename(
#@        columns={
#@            "effect_coef": "placebo_coef",
#@            "effect_p": "placebo_p",
#@        }
#@    )
#@    return stability.merge(
#@        placebo,
#@        on=["crisis", "treatment", "topic", "side"],
#@        how="left",
#@    )
#@
#@
#@def save_per_check_results(results: pd.DataFrame, output_dir: Path) -> None:
#@    folder_map = {
#@        "weighted": output_dir / "weighted",
#@        "cluster_pair": output_dir / "cluster_robustness" / "pair",
#@        "cluster_origin_mentioned": output_dir / "cluster_robustness" / "origin_mentioned",
#@        "cluster_origin": output_dir / "cluster_robustness" / "origin",
#@        "cluster_mentioned": output_dir / "cluster_robustness" / "mentioned",
#@        "raw_treatment": output_dir / "functional_form_raw_level",
#@        "placebo": output_dir / "placebo",
#@        "threshold_plus20": output_dir / "threshold_plus20",
#@    }
#@    for check_name, check_group in results.groupby("check_name", sort=False):
#@        folder = folder_map[str(check_name)]
#@        folder.mkdir(parents=True, exist_ok=True)
#@        for crisis, crisis_group in check_group.groupby("crisis", sort=False):
#@            crisis_group.to_csv(
#@                folder / f"results_{crisis}.csv",
#@                index=False,
#@                encoding="utf-8-sig",
#@            )
#@
#@
#@def main() -> None:
#@    args = parse_args()
#@    core.check_input_files(args.topic_file, args.asylum_file)
#@    args.output_dir.mkdir(parents=True, exist_ok=True)
#@
#@    print("Reading topic-labelled data...")
#@    topic_data = core.read_topic_labeled_data(args.topic_file, args.include_self)
#@    print(
#@        f"Usable labelled rows: {len(topic_data):,}; "
#@        f"origins={topic_data['origin'].nunique()}; "
#@        f"mentioned={topic_data['mentioned'].nunique()}"
#@    )
#@
#@    standard = run_standard_checks(
#@        topic_data,
#@        args.asylum_file,
#@        args.min_count_2015,
#@        args.min_count_2022,
#@    )
#@    placebo = run_placebo_checks(
#@        topic_data,
#@        args.asylum_file,
#@        args.min_count_2015,
#@        args.min_count_2022,
#@    )
#@    result_parts = standard[0] + placebo[0]
#@    sample_rows = standard[1] + placebo[1]
#@    unmatched_parts = standard[2] + placebo[2]
#@    construction_rows = standard[3] + placebo[3]
#@    placebo_attempts = placebo[4]
#@
#@    results = pd.concat(result_parts, ignore_index=True)
#@    concise = build_concise_summary(results)
#@    stability = build_stability_summary(concise)
#@    significant = concise.loc[
#@        concise["final_model_significant"]
#@    ].copy()
#@    failures = results.loc[
#@        (results["model_status"] != "success")
#@        | (results["term_status"] != "success")
#@    ].copy()
#@    unmatched = (
#@        pd.concat(unmatched_parts, ignore_index=True)
#@        if unmatched_parts
#@        else pd.DataFrame(
#@            columns=[
#@                "crisis", "treatment", "stage", "side", "unmatched_country", "check_name"
#@            ]
#@        )
#@    )
#@
#@    results.to_csv(
#@        args.output_dir / "topic_robustness_all_results.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    concise.to_csv(
#@        args.output_dir / "topic_robustness_concise_side_summary.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    stability.to_csv(
#@        args.output_dir / "topic_robustness_stability_summary.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    significant.to_csv(
#@        args.output_dir / "topic_robustness_significant_only.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    pd.DataFrame(sample_rows).to_csv(
#@        args.output_dir / "topic_robustness_sample_counts.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    pd.DataFrame(construction_rows).drop_duplicates().to_csv(
#@        args.output_dir / "topic_robustness_panel_construction.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    pd.DataFrame(placebo_attempts).to_csv(
#@        args.output_dir / "topic_robustness_placebo2015_attempts.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    failures.to_csv(
#@        args.output_dir / "topic_robustness_failures.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    unmatched.to_csv(
#@        args.output_dir / "topic_robustness_unmatched_countries.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    save_per_check_results(results, args.output_dir)
#@
#@    settings = {
#@        "topic_file": str(args.topic_file),
#@        "asylum_file": str(args.asylum_file),
#@        "output_dir": str(args.output_dir),
#@        "include_self": args.include_self,
#@        "base_min_count_2015": args.min_count_2015,
#@        "base_min_count_2022": args.min_count_2022,
#@        "threshold_plus20_2015": args.min_count_2015 + 20,
#@        "threshold_plus20_2022": args.min_count_2022 + 20,
#@        "primary_topics": core.PRIMARY_TOPICS,
#@        "denominator_topics": core.ALL_TOPICS,
#@        "checks": CHECK_ORDER,
#@        "model": (
#@            "PanelOLS with dyad and period fixed effects; origin and "
#@            "mentioned treatment terms estimated simultaneously"
#@        ),
#@        "weighted_model": (
#@            "PanelOLS with period-specific topic-count observation weights; "
#@            "w0*w1/(w0+w1) retained only as the profiled dyad-weight diagnostic"
#@        ),
#@        "cluster_covariance": (
#@            "PanelOLS clustered covariance; main/pair check clusters by dyad; "
#@            "two-way check clusters jointly by origin and mentioned country"
#@        ),
#@        "main_treatment_transform": "log1p",
#@        "reported_effect_scale": (
#@            "sample-average discrete effect of +1 applicant per 1,000 "
#@            "residents; raw-level alternative is already on this unit"
#@        ),
#@        "functional_form_check": "raw treatment level instead of log1p",
#@        "significance_rule": "pretrend p >= 0.05 and DID p < 0.05",
#@        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
#@    }
#@    (args.output_dir / "topic_robustness_settings.json").write_text(
#@        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
#@    )
#@
#@    print("\nFinished robustness checks.")
#@    print(f"Output directory: {args.output_dir}")
#@    print(f"All coefficient rows: {len(results):,}")
#@    print(f"Concise side rows: {len(concise):,}")
#@    print(f"Flagged result rows: {len(significant):,}")
#@    print(f"Failure/invalid-covariance rows: {len(failures):,}")
#@
#@
#@if __name__ == "__main__":
#@    main()
# === END EMBEDDED topic_checks ===


# === BEGIN EMBEDDED topic_dyads ===
#@"""
#@Sensitivity 1 and leave-one-country-out checks for the latest topic-share DID.
#@
#@This script imports every panel-construction, treatment, estimation, inference,
#@and reporting helper from ``topic_share_did_log1p_aligned.py``.  It changes only
#@the analysis sample:
#@
#@* Full sample: the unchanged benchmark from the latest main topic DID.
#@* Sensitivity 1: exclude Russia->Ukraine and Ukraine->Russia.
#@* Leave one country out: in turn, exclude every dyad in which the selected
#@  country is either the origin or the mentioned country.
#@
#@No source workbook is modified.  All outputs are written to a new directory.
#@"""
#@
#@from __future__ import annotations
#@
#@import argparse
#@import inspect
#@import json
#@from datetime import datetime
#@from pathlib import Path
#@
#@import numpy as np
#@import pandas as pd
#@
#@import topic_share_did_log1p_aligned as core
#@
#@
#@SAMPLE_FULL = "full_sample"
#@SAMPLE_S1 = "S1_exclude_direct_Russia_Ukraine_dyads"
#@SAMPLE_LOO = "leave_one_country_out"
#@
#@RUSSIA_NAMES = {"russia", "russian federation"}
#@UKRAINE_NAMES = {"ukraine"}
#@
#@DEFAULT_OUTPUT_DIR = (
#@    Path(__file__).resolve().parent / "topic_sensitivity_S1_LOO_results"
#@)
#@
#@
#@def validate_latest_core() -> None:
#@    """Fail early if an older topic-DID module is imported by mistake."""
#@    required_attributes = [
#@        "PanelOLS",
#@        "panel_ols_two_sided",
#@        "TREATMENT_TRANSFORM",
#@        "REPORT_AVERAGE_DISCRETE_EFFECT",
#@        "DISCRETE_INCREASE_PER_1000",
#@    ]
#@    missing = [name for name in required_attributes if not hasattr(core, name)]
#@    if missing:
#@        raise RuntimeError(
#@            "The imported topic_share_did_log1p_aligned.py is not the latest "
#@            f"PanelOLS version. Missing: {', '.join(missing)}. Imported from: "
#@            f"{Path(core.__file__).resolve()}"
#@        )
#@
#@    if core.TREATMENT_TRANSFORM != "log1p":
#@        raise RuntimeError(
#@            "The latest sensitivity code requires TREATMENT_TRANSFORM='log1p'; "
#@            f"received {core.TREATMENT_TRANSFORM!r} from "
#@            f"{Path(core.__file__).resolve()}"
#@        )
#@
#@    estimator_source = inspect.getsource(core.panel_ols_two_sided)
#@    required_model_settings = [
#@        "PanelOLS(",
#@        "entity_effects=True",
#@        "time_effects=True",
#@        'cov_type="clustered"',
#@        "cluster_entity=True",
#@    ]
#@    missing_settings = [
#@        setting for setting in required_model_settings
#@        if setting not in estimator_source
#@    ]
#@    if missing_settings:
#@        raise RuntimeError(
#@            "The imported core estimator is not the required PanelOLS model "
#@            "with dyad fixed effects, period fixed effects, and dyad-clustered "
#@            f"standard errors. Missing settings: {missing_settings}. Imported "
#@            f"from: {Path(core.__file__).resolve()}"
#@        )
#@
#@
#@def parse_args() -> argparse.Namespace:
#@    parser = argparse.ArgumentParser(description=__doc__)
#@    parser.add_argument("--topic-file", type=Path, default=core.DEFAULT_TOPIC_FILE)
#@    parser.add_argument("--asylum-file", type=Path, default=core.DEFAULT_ASYLUM_FILE)
#@    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
#@    parser.add_argument("--min-count-2015", type=int, default=core.MIN_COUNT_2015)
#@    parser.add_argument("--min-count-2022", type=int, default=core.MIN_COUNT_2022)
#@    parser.add_argument("--include-self", action="store_true")
#@    args, _notebook_arguments = parser.parse_known_args()
#@    return args
#@
#@
#@def _is_russia(series: pd.Series) -> pd.Series:
#@    return series.astype(str).str.lower().isin(RUSSIA_NAMES)
#@
#@
#@def _is_ukraine(series: pd.Series) -> pd.Series:
#@    return series.astype(str).str.lower().isin(UKRAINE_NAMES)
#@
#@
#@def filter_panel(
#@    panel: pd.DataFrame,
#@    sample_spec: str,
#@    omitted_country: str | None = None,
#@) -> tuple[pd.DataFrame, list[str]]:
#@    """Filter complete dyads after the latest main-code panel is constructed."""
#@    before_pairs = set(panel["pair"].dropna().astype(str))
#@
#@    if sample_spec == SAMPLE_FULL:
#@        mask = pd.Series(False, index=panel.index)
#@    elif sample_spec == SAMPLE_S1:
#@        mask = (
#@            (_is_russia(panel["origin"]) & _is_ukraine(panel["mentioned"]))
#@            | (_is_ukraine(panel["origin"]) & _is_russia(panel["mentioned"]))
#@        )
#@    elif sample_spec == SAMPLE_LOO:
#@        if omitted_country is None:
#@            raise ValueError("Leave-one-country-out requires omitted_country")
#@        omitted = core.clean_country_name(omitted_country)
#@        mask = (panel["origin"] == omitted) | (panel["mentioned"] == omitted)
#@    else:
#@        raise ValueError(f"Unknown sample specification: {sample_spec}")
#@
#@    filtered = panel.loc[~mask].copy().reset_index(drop=True)
#@    after_pairs = set(filtered["pair"].dropna().astype(str))
#@    return filtered, sorted(before_pairs - after_pairs)
#@
#@
#@def build_panels(topic_data: pd.DataFrame, args: argparse.Namespace):
#@    definitions = {
#@        "2015_pretrend": (
#@            {
#@                2010: "2010-2012",
#@                2011: "2010-2012",
#@                2012: "2010-2012",
#@                2013: "2013-2014",
#@                2014: "2013-2014",
#@            },
#@            args.min_count_2015,
#@        ),
#@        "2015_main": (
#@            {
#@                2013: "2013-2014",
#@                2014: "2013-2014",
#@                **{year: "2015-2021" for year in range(2015, 2022)},
#@            },
#@            args.min_count_2015,
#@        ),
#@        "2022_pretrend": (
#@            {
#@                **{year: "2010-2014" for year in range(2010, 2015)},
#@                **{year: "2015-2021" for year in range(2015, 2022)},
#@            },
#@            args.min_count_2022,
#@        ),
#@        "2022_main": (
#@            {
#@                **{year: "2015-2021" for year in range(2015, 2022)},
#@                **{year: "2022-2024" for year in range(2022, 2025)},
#@            },
#@            args.min_count_2022,
#@        ),
#@    }
#@
#@    panels = {}
#@    diagnostics = []
#@    for name, (mapping, minimum) in definitions.items():
#@        panel, diagnostic = core.build_topic_share_panel(
#@            topic_data, mapping, minimum, name
#@        )
#@        panels[name] = panel
#@        diagnostics.append(diagnostic)
#@    return panels, pd.DataFrame(diagnostics)
#@
#@
#@def comparison_definitions(
#@    panels: dict[str, pd.DataFrame], asylum_file: Path
#@) -> list[dict[str, object]]:
#@    treatment_2015 = core.load_treatment(asylum_file, 2015, "first_time_applicants")
#@    definitions = [
#@        {
#@            "crisis": "2015",
#@            "treatment": "first_time_applicants",
#@            "treatment_frame": treatment_2015,
#@            "pre_panel": panels["2015_pretrend"],
#@            "pre_baseline": "2013-2014",
#@            "pre_comparison": "2010-2012",
#@            "main_panel": panels["2015_main"],
#@            "main_baseline": "2013-2014",
#@            "main_comparison": "2015-2021",
#@        }
#@    ]
#@    for treatment_name in ["non-Ukrainian_first_time_applicant", "first_time_applicants_ukrainian"]:
#@        definitions.append(
#@            {
#@                "crisis": "2022",
#@                "treatment": treatment_name,
#@                "treatment_frame": core.load_treatment(
#@                    asylum_file, 2022, treatment_name
#@                ),
#@                "pre_panel": panels["2022_pretrend"],
#@                "pre_baseline": "2015-2021",
#@                "pre_comparison": "2010-2014",
#@                "main_panel": panels["2022_main"],
#@                "main_baseline": "2015-2021",
#@                "main_comparison": "2022-2024",
#@            }
#@        )
#@    return definitions
#@
#@
#@def run_stage(
#@    definition: dict[str, object],
#@    stage: str,
#@    sample_spec: str,
#@    omitted_country: str | None,
#@):
#@    panel = definition[f"{stage}_panel"]
#@    filtered, removed_pairs = filter_panel(panel, sample_spec, omitted_country)
#@    results, changes, unmatched, sample = core.run_comparison(
#@        filtered,
#@        definition["treatment_frame"],
#@        definition[f"{stage}_baseline"],
#@        definition[f"{stage}_comparison"],
#@        definition["crisis"],
#@        definition["treatment"],
#@        "pretrend" if stage == "pre" else "main_did",
#@    )
#@    results.insert(3, "sample_spec", sample_spec)
#@    results.insert(4, "omitted_country", omitted_country)
#@
#@    audit = {
#@        "crisis": definition["crisis"],
#@        "treatment": definition["treatment"],
#@        "stage": "pretrend" if stage == "pre" else "main_did",
#@        "sample_spec": sample_spec,
#@        "omitted_country": omitted_country,
#@        "pairs_before_filter": int(panel["pair"].nunique()),
#@        "rows_before_filter": int(len(panel)),
#@        "pairs_after_filter": int(filtered["pair"].nunique()),
#@        "rows_after_filter": int(len(filtered)),
#@        "pairs_removed_by_filter": int(len(removed_pairs)),
#@        "rows_removed_by_filter": int(len(panel) - len(filtered)),
#@        "removed_pairs": " | ".join(removed_pairs),
#@        "pairs_after_treatment_matching": sample[
#@            "pairs_after_both_treatments_matched"
#@        ],
#@        "n_origins_after_matching": sample["n_origins"],
#@        "n_mentioned_after_matching": sample["n_mentioned"],
#@        "unmatched_country_records": int(len(unmatched)),
#@        "status": "ok",
#@    }
#@    return results, changes, audit
#@
#@
#@def scenario_summary_long(
#@    raw_results: pd.DataFrame,
#@    sample_spec: str,
#@    omitted_country: str | None,
#@) -> pd.DataFrame:
#@    wide = core.build_main_summary(raw_results)
#@    rows = []
#@    for _, row in wide.iterrows():
#@        for side in ["origin", "mentioned"]:
#@            rows.append(
#@                {
#@                    "crisis": row["crisis"],
#@                    "treatment": row["treatment"],
#@                    "topic": row["topic"],
#@                    "side": side,
#@                    "sample_spec": sample_spec,
#@                    "omitted_country": omitted_country,
#@                    "pretrend_coef": row[f"{side}_pretrend_coef"],
#@                    "pretrend_se": row[f"{side}_pretrend_se"],
#@                    "pretrend_p": row[f"{side}_pretrend_p"],
#@                    "pretrend_ci_low": row[f"{side}_pretrend_ci_low"],
#@                    "pretrend_ci_high": row[f"{side}_pretrend_ci_high"],
#@                    "main_did_coef": row[f"{side}_main_did_coef"],
#@                    "main_did_se": row[f"{side}_main_did_se"],
#@                    "main_did_p": row[f"{side}_main_did_p"],
#@                    "main_did_ci_low": row[f"{side}_main_did_ci_low"],
#@                    "main_did_ci_high": row[f"{side}_main_did_ci_high"],
#@                    "main_did_model_log1p_coef": row[
#@                        f"{side}_main_did_model_coef"
#@                    ],
#@                    "average_discrete_factor": row[
#@                        f"{side}_main_did_average_discrete_factor"
#@                    ],
#@                    "pretrend_n_pairs": row[f"{side}_pretrend_n_pairs"],
#@                    "main_did_n_pairs": row[f"{side}_main_did_n_pairs"],
#@                    "parallel_trend_pass": row[
#@                        f"{side}_parallel_trend_pass"
#@                    ],
#@                    "main_did_significant": row[
#@                        f"{side}_main_did_significant"
#@                    ],
#@                    "final_significant": row[f"{side}_final_significant"],
#@                }
#@            )
#@    return pd.DataFrame(rows)
#@
#@
#@def country_universe(*changes: pd.DataFrame) -> list[str]:
#@    countries: set[str] = set()
#@    for frame in changes:
#@        countries.update(frame["origin"].dropna().astype(str))
#@        countries.update(frame["mentioned"].dropna().astype(str))
#@    return sorted(countries)
#@
#@
#@def run_scenario(
#@    definition: dict[str, object],
#@    sample_spec: str,
#@    omitted_country: str | None,
#@):
#@    pre_results, pre_changes, pre_audit = run_stage(
#@        definition, "pre", sample_spec, omitted_country
#@    )
#@    main_results, main_changes, main_audit = run_stage(
#@        definition, "main", sample_spec, omitted_country
#@    )
#@    raw = pd.concat([pre_results, main_results], ignore_index=True)
#@    summary = scenario_summary_long(raw, sample_spec, omitted_country)
#@    return raw, summary, pre_changes, main_changes, [pre_audit, main_audit]
#@
#@
#@def build_stability_summary(summary: pd.DataFrame) -> pd.DataFrame:
#@    keys = ["crisis", "treatment", "topic", "side"]
#@    full = summary.loc[summary["sample_spec"] == SAMPLE_FULL].copy()
#@    s1 = summary.loc[summary["sample_spec"] == SAMPLE_S1].copy()
#@    loo = summary.loc[summary["sample_spec"] == SAMPLE_LOO].copy()
#@
#@    s1_columns = keys + [
#@        "pretrend_p",
#@        "main_did_coef",
#@        "main_did_se",
#@        "main_did_p",
#@        "main_did_ci_low",
#@        "main_did_ci_high",
#@        "parallel_trend_pass",
#@        "main_did_significant",
#@        "final_significant",
#@    ]
#@    s1 = s1[s1_columns].rename(
#@        columns={column: f"s1_{column}" for column in s1_columns if column not in keys}
#@    )
#@    merged = full.merge(s1, on=keys, how="left", validate="one_to_one")
#@
#@    rows = []
#@    for _, base in merged.iterrows():
#@        group = loo.copy()
#@        for key in keys:
#@            group = group.loc[group[key] == base[key]]
#@        group = group.loc[group["main_did_coef"].notna()].copy()
#@        full_coef = float(base["main_did_coef"])
#@        group["coef_change_from_full"] = group["main_did_coef"] - full_coef
#@        group["abs_coef_change_from_full"] = group[
#@            "coef_change_from_full"
#@        ].abs()
#@        group["same_sign_as_full"] = np.sign(group["main_did_coef"]) == np.sign(
#@            full_coef
#@        )
#@        group["main_ci_excludes_zero"] = (
#@            group["main_did_ci_low"] * group["main_did_ci_high"] > 0
#@        )
#@        influential = (
#@            group.sort_values(
#@                ["abs_coef_change_from_full", "omitted_country"],
#@                ascending=[False, True],
#@            ).iloc[0]
#@            if not group.empty
#@            else None
#@        )
#@        rows.append(
#@            {
#@                "crisis": base["crisis"],
#@                "treatment": base["treatment"],
#@                "topic": base["topic"],
#@                "side": base["side"],
#@                "full_pretrend_p": base["pretrend_p"],
#@                "full_main_did_coef": full_coef,
#@                "full_main_did_se": base["main_did_se"],
#@                "full_main_did_p": base["main_did_p"],
#@                "full_main_did_ci_low": base["main_did_ci_low"],
#@                "full_main_did_ci_high": base["main_did_ci_high"],
#@                "full_parallel_trend_pass": base["parallel_trend_pass"],
#@                "full_main_did_significant": base["main_did_significant"],
#@                "full_final_significant": base["final_significant"],
#@                "s1_pretrend_p": base["s1_pretrend_p"],
#@                "s1_main_did_coef": base["s1_main_did_coef"],
#@                "s1_main_did_se": base["s1_main_did_se"],
#@                "s1_main_did_p": base["s1_main_did_p"],
#@                "s1_main_did_ci_low": base["s1_main_did_ci_low"],
#@                "s1_main_did_ci_high": base["s1_main_did_ci_high"],
#@                "s1_parallel_trend_pass": base["s1_parallel_trend_pass"],
#@                "s1_main_did_significant": base["s1_main_did_significant"],
#@                "s1_final_significant": base["s1_final_significant"],
#@                "loo_models": int(len(group)),
#@                "loo_min_coef": group["main_did_coef"].min(),
#@                "loo_max_coef": group["main_did_coef"].max(),
#@                "loo_same_sign_models": int(group["same_sign_as_full"].sum()),
#@                "loo_main_p_lt_0_05": int(group["main_did_significant"].sum()),
#@                "loo_final_significant": int(group["final_significant"].sum()),
#@                "loo_main_ci_excludes_zero": int(
#@                    group["main_ci_excludes_zero"].sum()
#@                ),
#@                "most_influential_omitted_country": (
#@                    influential["omitted_country"] if influential is not None else None
#@                ),
#@                "largest_absolute_coef_change": (
#@                    influential["abs_coef_change_from_full"]
#@                    if influential is not None
#@                    else np.nan
#@                ),
#@                "coef_when_most_influential_omitted": (
#@                    influential["main_did_coef"]
#@                    if influential is not None
#@                    else np.nan
#@                ),
#@                "p_when_most_influential_omitted": (
#@                    influential["main_did_p"]
#@                    if influential is not None
#@                    else np.nan
#@                ),
#@            }
#@        )
#@    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
#@
#@
#@def add_loo_changes(summary: pd.DataFrame) -> pd.DataFrame:
#@    keys = ["crisis", "treatment", "topic", "side"]
#@    full = summary.loc[summary["sample_spec"] == SAMPLE_FULL, keys + [
#@        "main_did_coef", "main_did_p", "final_significant"
#@    ]].rename(
#@        columns={
#@            "main_did_coef": "full_main_did_coef",
#@            "main_did_p": "full_main_did_p",
#@            "final_significant": "full_final_significant",
#@        }
#@    )
#@    loo = summary.loc[summary["sample_spec"] == SAMPLE_LOO].merge(
#@        full, on=keys, how="left", validate="many_to_one"
#@    )
#@    loo["coef_change_from_full"] = (
#@        loo["main_did_coef"] - loo["full_main_did_coef"]
#@    )
#@    loo["absolute_coef_change"] = loo["coef_change_from_full"].abs()
#@    loo["same_sign_as_full"] = (
#@        np.sign(loo["main_did_coef"]) == np.sign(loo["full_main_did_coef"])
#@    )
#@    loo["main_ci_excludes_zero"] = (
#@        loo["main_did_ci_low"] * loo["main_did_ci_high"] > 0
#@    )
#@    return loo.sort_values(keys + ["omitted_country"]).reset_index(drop=True)
#@
#@
#@def main() -> None:
#@    args = parse_args()
#@    validate_latest_core()
#@    core.check_input_files(args.topic_file, args.asylum_file)
#@    args.output_dir.mkdir(parents=True, exist_ok=True)
#@
#@    print("Reading topic-labelled data...")
#@    topic_data = core.read_topic_labeled_data(args.topic_file, args.include_self)
#@    print(f"Usable labelled rows: {len(topic_data):,}")
#@    panels, construction = build_panels(topic_data, args)
#@    definitions = comparison_definitions(panels, args.asylum_file)
#@
#@    raw_parts = []
#@    summary_parts = []
#@    audit_rows = []
#@
#@    for definition in definitions:
#@        label = f"{definition['crisis']} / {definition['treatment']}"
#@        print(f"Running full benchmark: {label}")
#@        raw, summary, pre_changes, main_changes, audit = run_scenario(
#@            definition, SAMPLE_FULL, None
#@        )
#@        raw_parts.append(raw)
#@        summary_parts.append(summary)
#@        audit_rows.extend(audit)
#@        countries = country_universe(pre_changes, main_changes)
#@
#@        print(f"Running Sensitivity 1: {label}")
#@        raw, summary, _pre, _main, audit = run_scenario(
#@            definition, SAMPLE_S1, None
#@        )
#@        raw_parts.append(raw)
#@        summary_parts.append(summary)
#@        audit_rows.extend(audit)
#@
#@        for index, country in enumerate(countries, start=1):
#@            print(f"LOO {label}: {index}/{len(countries)} {country}")
#@            raw, summary, _pre, _main, audit = run_scenario(
#@                definition, SAMPLE_LOO, country
#@            )
#@            raw_parts.append(raw)
#@            summary_parts.append(summary)
#@            audit_rows.extend(audit)
#@
#@    raw_results = pd.concat(raw_parts, ignore_index=True)
#@    scenario_summary = pd.concat(summary_parts, ignore_index=True)
#@    sample_audit = pd.DataFrame(audit_rows)
#@    stability = build_stability_summary(scenario_summary)
#@    loo_detail = add_loo_changes(scenario_summary)
#@    significant_readout = stability.loc[
#@        stability["full_final_significant"].astype(bool)
#@    ].copy()
#@
#@    raw_results.to_csv(
#@        args.output_dir / "topic_sensitivity_all_coefficients.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    scenario_summary.to_csv(
#@        args.output_dir / "topic_sensitivity_scenario_summary.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    loo_detail.to_csv(
#@        args.output_dir / "topic_sensitivity_leave_one_country_detail.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    stability.to_csv(
#@        args.output_dir / "topic_sensitivity_stability_summary.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    significant_readout.to_csv(
#@        args.output_dir / "topic_sensitivity_main_significant_readout.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    sample_audit.to_csv(
#@        args.output_dir / "topic_sensitivity_sample_audit.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@    construction.to_csv(
#@        args.output_dir / "topic_sensitivity_panel_construction.csv",
#@        index=False,
#@        encoding="utf-8-sig",
#@    )
#@
#@    settings = {
#@        "topic_file": str(args.topic_file),
#@        "asylum_file": str(args.asylum_file),
#@        "main_code": str(Path(core.__file__).resolve()),
#@        "include_self": bool(args.include_self),
#@        "min_count_2015": int(args.min_count_2015),
#@        "min_count_2022": int(args.min_count_2022),
#@        "topic_denominator": " | ".join(core.ALL_TOPICS),
#@        "primary_topics": " | ".join(core.PRIMARY_TOPICS),
#@        "treatment_transform": core.TREATMENT_TRANSFORM,
#@        "reported_effect": (
#@            "sample-average discrete effect of +1 applicant per 1,000 residents"
#@        ),
#@        "estimator": (
#@            "PanelOLS with dyad fixed effects and period fixed effects; "
#@            "origin-side and mentioned-side post x log1p treatment terms "
#@            "included jointly"
#@        ),
#@        "model_treatment_scale": (
#@            "log(1 + first-time asylum applicants per 1,000 residents)"
#@        ),
#@        "covariance": "standard errors clustered by dyad (PanelOLS cluster_entity=True)",
#@        "sensitivity_1": (
#@            "exclude Russia->Ukraine and Ukraine->Russia if present in the "
#@            "thresholded balanced analysis panel"
#@        ),
#@        "leave_one_country_out": (
#@            "omit all dyads involving each country as origin or mentioned, one at a time"
#@        ),
#@        "significance_rule": "pretrend p >= 0.05 and main DID p < 0.05",
#@        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
#@    }
#@    (args.output_dir / "topic_sensitivity_settings.json").write_text(
#@        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
#@    )
#@
#@    print("Finished.")
#@    print(f"Output directory: {args.output_dir}")
#@    print(
#@        significant_readout[
#@            [
#@                "crisis",
#@                "treatment",
#@                "topic",
#@                "side",
#@                "full_main_did_coef",
#@                "full_main_did_p",
#@                "s1_main_did_coef",
#@                "s1_main_did_p",
#@                "s1_final_significant",
#@                "loo_models",
#@                "loo_same_sign_models",
#@                "loo_final_significant",
#@                "most_influential_omitted_country",
#@            ]
#@        ].to_string(index=False)
#@    )
#@
#@
#@if __name__ == "__main__":
#@    main()
# === END EMBEDDED topic_dyads ===
