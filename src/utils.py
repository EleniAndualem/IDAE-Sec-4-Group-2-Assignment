# Load and explore housing data

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "California Housing Prices.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_understanding.md"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "median_house_value"


def ensure_output_dirs():
    """Ensure output directories exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    # Read CSV into a DataFrame
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> dict:
    # Check size, types, missing values, duplicates
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def descriptive_statistics(df: pd.DataFrame) -> dict:
    # Get summary stats for numbers and categories
    numeric_stats = df.describe().to_dict()
    categorical_stats = {
        col: df[col].value_counts().to_dict()
        for col in df.select_dtypes(include=["object", "string"]).columns
    }
    return {"numeric": numeric_stats, "categorical": categorical_stats}


def build_report_markdown(inspection: dict, stats: dict) -> str:
    # Turn results into markdown text
    rows, cols = inspection["shape"]
    lines = [
        "# Data Understanding Report",
        "",
        "## Dataset Overview",
        f"- **Rows:** {rows:,}",
        f"- **Columns:** {cols}",
        f"- **Duplicate rows:** {inspection['duplicate_rows']}",
        "",
        "## Columns",
        "",
        "| Column | Data Type | Missing Values |",
        "|--------|-----------|----------------|",
    ]

    for col in inspection["columns"]:
        dtype = inspection["dtypes"][col]
        missing = inspection["missing_values"][col]
        lines.append(f"| {col} | {dtype} | {missing} |")

    lines.extend(["", "## Numeric Statistics", ""])
    for col, values in stats["numeric"].items():
        lines.append(f"### {col}")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|------|-------|")
        for stat_name, value in values.items():
            formatted = f"{value:,.4f}" if isinstance(value, float) else value
            lines.append(f"| {stat_name} | {formatted} |")
        lines.append("")

    lines.append("## Categorical Statistics")
    lines.append("")
    for col, counts in stats["categorical"].items():
        lines.append(f"### {col}")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for category, count in counts.items():
            lines.append(f"| {category} | {count:,} |")
        lines.append("")

    return "\n".join(lines)


def save_data_understanding_report(
    csv_path: Path | str = DEFAULT_DATA_PATH,
    output_path: Path | str = DEFAULT_REPORT_PATH,
) -> Path:
    # Save report to outputs/reports/
    df = load_data(csv_path)
    inspection = inspect_data(df)
    stats = descriptive_statistics(df)
    report = build_report_markdown(inspection, stats)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def save_markdown(content: str, path: Path | str) -> Path:
    # Write markdown string to a file
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


if __name__ == "__main__":
    report_path = save_data_understanding_report()
    print(f"Report saved to: {report_path}")
