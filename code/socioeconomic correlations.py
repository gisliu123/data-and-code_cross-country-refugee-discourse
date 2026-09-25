"""Run the combined country-level socioeconomic regression analysis.

Usage:
    python run_analysis.py

The script reads the three standardized CSV files in the adjacent ``data``
folder. It creates a ``results`` folder containing only full-model regression
tables, analysis datasets, data-matching reports, and observation counts.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "openpyxl": "openpyxl",
}


def install_missing_packages() -> None:
    """Install packages that are not available in the active Python environment."""
    missing_packages = []

    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if not missing_packages:
        return

    print("Installing required Python packages:", ", ".join(missing_packages))
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing_packages]
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Required Python packages could not be installed. "
            "Check the internet connection and run the script again."
        ) from error


install_missing_packages()

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill
from scipy import stats


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"

DATA_FILE_NAMES = {
    "socioeconomic_indicators": "socioeconomic_indicators.csv",
    "user_country_outcomes": "user_country_outcomes.csv",
    "mentioned_country_outcomes": "mentioned_country_outcomes.csv",
}

PERIODS = {
    "2010_2014": range(2010, 2015),
    "2015_2021": range(2015, 2022),
    "2022_2024": range(2022, 2025),
}

FULL_MODEL_PREDICTORS = [
    "log_gdp_per_capita",
    "log_unemployment_rate",
    "log_refugee_proportion",
    "log_tertiary_enrollment_rate",
]

REGRESSION_RESULT_COLUMNS = [
    "term",
    "estimate",
    "standard_error",
    "t_statistic",
    "p_value",
]

def normalize_country_name(country_series: pd.Series) -> pd.Series:
    """Create a case-insensitive country key without changing display names."""
    return country_series.fillna("").astype(str).str.strip().str.casefold()


def require_columns(
    data_frame: pd.DataFrame,
    required_columns: list[str],
    file_path: Path,
) -> None:
    """Raise a clear error when a required column is missing."""
    missing_columns = sorted(set(required_columns) - set(data_frame.columns))
    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path.name}: {', '.join(missing_columns)}"
        )


def validate_unique_country_year(
    data_frame: pd.DataFrame,
    file_path: Path,
) -> None:
    """Require exactly one record per country and year."""
    duplicate_mask = data_frame.duplicated(
        subset=["country_key", "year"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"{file_path.name} contains {duplicate_count} duplicate "
            "country-year rows."
        )


def read_standardized_csv(
    file_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    """Read and validate one standardized input file."""
    file_path = DATA_DIRECTORY / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file not found: {file_path}")

    data_frame = pd.read_csv(file_path)
    require_columns(data_frame, required_columns, file_path)

    data_frame["country"] = data_frame["country"].fillna("").astype(str).str.strip()
    data_frame["country_key"] = normalize_country_name(data_frame["country"])
    data_frame["year"] = pd.to_numeric(
        data_frame["year"], errors="raise"
    ).astype(int)

    if (data_frame["country_key"] == "").any():
        raise ValueError(f"{file_path.name} contains a blank country name.")

    validate_unique_country_year(data_frame, file_path)
    return data_frame


def load_input_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the socioeconomic and outcome data files."""
    socioeconomic_columns = [
        "country",
        "year",
        "gdp_per_capita",
        "unemployment_rate",
        "tertiary_enrollment_rate",
        "refugee_population",
        "total_population",
    ]
    outcome_columns = [
        "country",
        "year",
        "sentiment_score",
        "hate_speech_score",
    ]

    socioeconomic_data = read_standardized_csv(
        DATA_FILE_NAMES["socioeconomic_indicators"],
        socioeconomic_columns,
    )
    user_country_outcomes = read_standardized_csv(
        DATA_FILE_NAMES["user_country_outcomes"],
        outcome_columns,
    )
    mentioned_country_outcomes = read_standardized_csv(
        DATA_FILE_NAMES["mentioned_country_outcomes"],
        outcome_columns,
    )

    numeric_columns = socioeconomic_columns[2:]
    for column_name in numeric_columns:
        socioeconomic_data[column_name] = pd.to_numeric(
            socioeconomic_data[column_name],
            errors="coerce",
        )

    for outcome_data in (user_country_outcomes, mentioned_country_outcomes):
        outcome_data["sentiment_score"] = pd.to_numeric(
            outcome_data["sentiment_score"],
            errors="coerce",
        )
        outcome_data["hate_speech_score"] = pd.to_numeric(
            outcome_data["hate_speech_score"],
            errors="coerce",
        )

    return (
        socioeconomic_data,
        user_country_outcomes,
        mentioned_country_outcomes,
    )


def write_excel_workbook(
    file_path: Path,
    worksheets: dict[str, pd.DataFrame],
) -> None:
    """Write data frames to a consistently formatted Excel workbook."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for worksheet_name, data_frame in worksheets.items():
            data_frame.to_excel(
                writer,
                sheet_name=worksheet_name,
                index=False,
            )

        workbook = writer.book
        header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
        header_font = Font(bold=True)

        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font

            for column_cells in worksheet.columns:
                maximum_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                adjusted_width = min(max(maximum_length + 2, 12), 40)
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = adjusted_width


def calculate_clustered_hc0_regression(
    analysis_data: pd.DataFrame,
    predictor_names: list[str],
) -> pd.DataFrame:
    """Fit OLS and calculate country-clustered HC0 standard errors."""
    response_values = analysis_data["outcome_value"].to_numpy(dtype=float)
    predictor_values = analysis_data[predictor_names].to_numpy(dtype=float)
    design_matrix = np.column_stack(
        [np.ones(len(analysis_data), dtype=float), predictor_values]
    )

    coefficient_values, _, matrix_rank, _ = np.linalg.lstsq(
        design_matrix,
        response_values,
        rcond=None,
    )

    parameter_count = design_matrix.shape[1]
    if matrix_rank < parameter_count:
        raise ValueError(
            "The regression design matrix is singular for predictors: "
            + ", ".join(predictor_names)
        )

    residual_degrees_of_freedom = len(analysis_data) - parameter_count
    if residual_degrees_of_freedom <= 0:
        raise ValueError("The regression has no residual degrees of freedom.")

    residual_values = response_values - design_matrix @ coefficient_values
    bread_matrix = np.linalg.inv(design_matrix.T @ design_matrix)
    score_matrix = design_matrix * residual_values[:, np.newaxis]

    country_groups = analysis_data["country_key"].to_numpy()
    unique_countries = np.unique(country_groups)
    country_count = len(unique_countries)

    if country_count < 2:
        raise ValueError(
            "Country-clustered standard errors require at least two countries."
        )

    meat_matrix = np.zeros((parameter_count, parameter_count), dtype=float)
    for country_name in unique_countries:
        country_score = score_matrix[country_groups == country_name].sum(axis=0)
        meat_matrix += np.outer(country_score, country_score)

    cluster_adjustment = country_count / (country_count - 1)
    covariance_matrix = (
        bread_matrix @ meat_matrix @ bread_matrix * cluster_adjustment
    )
    variance_values = np.maximum(np.diag(covariance_matrix), 0.0)
    standard_errors = np.sqrt(variance_values)

    with np.errstate(divide="ignore", invalid="ignore"):
        test_statistics = coefficient_values / standard_errors

    p_values = 2 * stats.t.sf(
        np.abs(test_statistics),
        df=residual_degrees_of_freedom,
    )

    return pd.DataFrame(
        {
            "term": ["intercept", *predictor_names],
            "estimate": coefficient_values,
            "standard_error": standard_errors,
            "t_statistic": test_statistics,
            "p_value": p_values,
        }
    )


def build_analysis_panel(
    socioeconomic_data: pd.DataFrame,
    outcome_data: pd.DataFrame,
    outcome_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match country-year records and create clearly named log variables."""
    available_keys = socioeconomic_data[["country_key", "year"]].drop_duplicates()
    match_check = outcome_data.merge(
        available_keys,
        on=["country_key", "year"],
        how="left",
        indicator=True,
    )
    unmatched_rows = match_check.loc[
        match_check["_merge"] == "left_only",
        ["country", "year", outcome_column],
    ].copy()

    match_summary = pd.DataFrame(
        {
            "metric": [
                "outcome_rows",
                "matched_country_year_rows",
                "unmatched_country_year_rows",
            ],
            "value": [
                len(outcome_data),
                int((match_check["_merge"] == "both").sum()),
                len(unmatched_rows),
            ],
        }
    )

    socioeconomic_values = socioeconomic_data.drop(columns=["country"])
    selected_outcomes = outcome_data[
        ["country", "country_key", "year", outcome_column]
    ]
    merged_data = socioeconomic_values.merge(
        selected_outcomes,
        on=["country_key", "year"],
        how="inner",
        validate="one_to_one",
    )

    merged_data["refugee_proportion"] = np.where(
        merged_data["total_population"] > 0,
        merged_data["refugee_population"] / merged_data["total_population"],
        np.nan,
    )

    logarithm_specifications = {
        "gdp_per_capita": "log_gdp_per_capita",
        "unemployment_rate": "log_unemployment_rate",
        "refugee_proportion": "log_refugee_proportion",
        "tertiary_enrollment_rate": "log_tertiary_enrollment_rate",
    }

    for source_column, logarithm_column in logarithm_specifications.items():
        merged_data[logarithm_column] = np.nan
        positive_mask = merged_data[source_column] > 0
        merged_data.loc[positive_mask, logarithm_column] = np.log(
            merged_data.loc[positive_mask, source_column]
        )

    analysis_data = merged_data[
        [
            "country",
            "country_key",
            "year",
            *FULL_MODEL_PREDICTORS,
            outcome_column,
        ]
    ].rename(columns={outcome_column: "outcome_value"})

    analysis_data = analysis_data.dropna(
        subset=[*FULL_MODEL_PREDICTORS, "outcome_value"]
    ).reset_index(drop=True)

    return analysis_data, match_summary, unmatched_rows


def run_single_analysis(
    analysis_identifier: str,
    analysis_side: str,
    outcome_name: str,
    outcome_column: str,
    outcome_data: pd.DataFrame,
    socioeconomic_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run all periods for one side and one outcome."""
    print(f"\nStarting {analysis_identifier}")
    analysis_output_directory = RESULTS_DIRECTORY / analysis_identifier
    analysis_output_directory.mkdir(parents=True, exist_ok=True)

    analysis_data, match_summary, unmatched_rows = build_analysis_panel(
        socioeconomic_data,
        outcome_data,
        outcome_column,
    )

    write_excel_workbook(
        analysis_output_directory / "data_match_report.xlsx",
        {
            "summary": match_summary,
            "unmatched_rows": unmatched_rows,
        },
    )

    full_regression_summaries = []
    observation_records = []

    for period_name, period_years in PERIODS.items():
        period_data = analysis_data[
            analysis_data["year"].isin(period_years)
        ].copy()
        observation_count = len(period_data)
        country_count = period_data["country_key"].nunique()

        if observation_count < 6:
            status = "skipped_fewer_than_6_observations"
        elif country_count < 2:
            status = "skipped_fewer_than_2_countries"
        else:
            status = "completed"

        observation_records.append(
            {
                "analysis": analysis_identifier,
                "outcome": outcome_name,
                "analysis_side": analysis_side,
                "period": period_name,
                "observation_count": observation_count,
                "country_count": country_count,
                "status": status,
            }
        )

        print(
            f"{period_name}: observations={observation_count}, "
            f"countries={country_count}, status={status}"
        )

        if status != "completed":
            continue

        full_regression_results = calculate_clustered_hc0_regression(
            period_data,
            FULL_MODEL_PREDICTORS,
        )

        period_output_directory = analysis_output_directory / period_name
        period_output_directory.mkdir(parents=True, exist_ok=True)

        export_data = period_data.drop(columns=["country_key"])
        write_excel_workbook(
            period_output_directory / "full_regression_results.xlsx",
            {"full_regression": full_regression_results},
        )
        write_excel_workbook(
            period_output_directory / "analysis_dataset.xlsx",
            {"analysis_data": export_data},
        )

        full_period_summary = full_regression_results.copy()
        full_period_summary.insert(0, "period", period_name)
        full_regression_summaries.append(full_period_summary)

    full_summary = (
        pd.concat(full_regression_summaries, ignore_index=True)
        if full_regression_summaries
        else pd.DataFrame(columns=["period", *REGRESSION_RESULT_COLUMNS])
    )

    write_excel_workbook(
        analysis_output_directory / "all_periods_full_regression.xlsx",
        {"full_regression": full_summary},
    )

    return pd.DataFrame(observation_records)


def main() -> None:
    """Load inputs, run four analyses, and write all results."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (
        socioeconomic_data,
        user_country_outcomes,
        mentioned_country_outcomes,
    ) = load_input_data()

    analysis_configurations = [
        {
            "analysis_identifier": "user_country_sentiment",
            "analysis_side": "user_country",
            "outcome_name": "sentiment",
            "outcome_column": "sentiment_score",
            "outcome_data": user_country_outcomes,
        },
        {
            "analysis_identifier": "user_country_hate_speech",
            "analysis_side": "user_country",
            "outcome_name": "hate_speech",
            "outcome_column": "hate_speech_score",
            "outcome_data": user_country_outcomes,
        },
        {
            "analysis_identifier": "mentioned_country_sentiment",
            "analysis_side": "mentioned_country",
            "outcome_name": "sentiment",
            "outcome_column": "sentiment_score",
            "outcome_data": mentioned_country_outcomes,
        },
        {
            "analysis_identifier": "mentioned_country_hate_speech",
            "analysis_side": "mentioned_country",
            "outcome_name": "hate_speech",
            "outcome_column": "hate_speech_score",
            "outcome_data": mentioned_country_outcomes,
        },
    ]

    observation_tables = []
    for configuration in analysis_configurations:
        observation_tables.append(
            run_single_analysis(
                socioeconomic_data=socioeconomic_data,
                **configuration,
            )
        )

    observation_counts = pd.concat(
        observation_tables,
        ignore_index=True,
    )
    write_excel_workbook(
        RESULTS_DIRECTORY / "observation_counts.xlsx",
        {"observation_counts": observation_counts},
    )

    print("\nAll analyses completed successfully.")
    print(f"Results directory: {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()

