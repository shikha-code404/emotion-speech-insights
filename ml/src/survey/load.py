"""Load and validate the DASS-21 survey workbook.

SYNTHETIC DATA: every respondent in survey_dataset_1200.xlsx is artificially
generated (see the workbook's "Notes" sheet). Nothing derived from it is a
real-world finding.

This module is the single source of truth for column names and the allowed
lifestyle options (Rule 3). Option strings keep the form's exact spellings,
including "Former (Quite)" and "Due to Anxity".
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = REPO_ROOT / "ml" / "data" / "raw" / "survey" / "survey_dataset_1200.xlsx"
SHEET = "Labeled_Dataset"
N_ITEMS = 21

SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA: all 1,200 survey respondents are artificially generated; "
    "none are real people. Results reflect the data generator's rules, not "
    "real-world patterns, and are not real-world accuracy."
)

# Source columns dropped on load: identifiers, free text, consent and time.
DROP_COLUMNS = (
    "Response_ID",
    "Timestamp",
    "Do you agree to participate in this survey?",
    "Name",
)

# Source header -> snake_case name. Q1..Q21 headers are matched by prefix.
RENAME = {
    "Gender": "gender",
    "Height (CM)": "height_cm",
    "Weight (kg)": "weight_kg",
    "Age Group": "age_group",
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
    "Current smoking status": "smoking_status",
    "Current alcohol consumption status": "alcohol_status",
    "Daily Step Count": "daily_steps",
    "Sleep Duration": "sleep_duration",
    "Daily Water Intake (liter)": "water_intake_l",
    "Time dedicated to physical activities": "physical_activity_time",
    "Do you tend to snack mindlessly?": "mindless_snacking",
    "Recurring physical discomforts (headaches, chest pain, etc.)": "physical_discomfort",
    "Occupation": "occupation",
    "Depression_Score": "depression_score",
    "Anxiety_Score": "anxiety_score",
    "Stress_Score": "stress_score",
    "Depression_Severity": "depression_severity",
    "Anxiety_Severity": "anxiety_severity",
    "Stress_Severity": "stress_severity",
    "Emotional_State": "emotional_state",
    "Severity_Level": "severity_level",
}

Q_COLUMNS = tuple(f"q{i}" for i in range(1, N_ITEMS + 1))
_Q_HEADER = re.compile(r"^Q(\d+)\.\s*(.+)$")

# Allowed options, in the form's order and exact spelling. Lists are ordered
# where the answer is ordinal (age_group, sleep_duration, ...).
CATEGORICAL_OPTIONS: dict[str, tuple[str, ...]] = {
    "gender": ("Male", "Female", "Prefer not to say"),
    "age_group": ("Below 18", "18 - 25", "26 - 35", "36 - 50", "50+"),
    "smoking_status": (
        "Never Smoked", "Former (Quit) smoker", "Current some day smoker",
        "Current every day smoker", "Prefer not to say",
    ),
    "alcohol_status": (
        "Never", "Former (Quite)", "Current some day", "Current every day",
        "Prefer not to say",
    ),
    "sleep_duration": (
        "2 - 3 hrs", "3 - 4 hrs", "4 - 5 hrs", "5 - 6 hrs", "6 - 7 hrs",
        "7 - 8 hrs", "8 - 9 hrs", "9 - 10 hrs", "10 - 11 hrs", "11 - 12 hrs",
    ),
    "physical_activity_time": (
        "Never", "Once in a while", "Weekly", "30 min - 1 hrs", "1 - 2 hrs",
    ),
    "mindless_snacking": ("Randomly", "Due to Happiness", "Due to Stress", "Due to Anxity"),
    "physical_discomfort": ("Never", "Once in a while", "Constantly"),
    "occupation": (
        "Student", "Working", "Business", "Self-employed", "Freelancer",
        "Farmer", "Homemaker", "Job Seeker", "Retired",
    ),
}

NUMERIC_FIELDS: dict[str, dict] = {
    "height_cm": {"unit": "cm"},
    "weight_kg": {"unit": "kg"},
    "systolic_bp": {"unit": "mmHg"},
    "diastolic_bp": {"unit": "mmHg"},
    "daily_steps": {"unit": "steps/day"},
    "water_intake_l": {"unit": "litres/day"},
}

# Score / label columns produced by DASS-21 scoring (never model inputs).
SCORE_COLUMNS = ("depression_score", "anxiety_score", "stress_score")
SEVERITY_COLUMNS = ("depression_severity", "anxiety_severity", "stress_severity")
LABEL_COLUMNS = ("emotional_state", "severity_level")


class SurveyValidationError(ValueError):
    pass


def item_texts(raw_columns) -> dict[str, str]:
    """{"q1": "I found it hard to wind down", ...} from the source headers."""
    out = {}
    for col in raw_columns:
        m = _Q_HEADER.match(str(col))
        if m:
            out[f"q{int(m.group(1))}"] = m.group(2).strip()
    return out


def _rename_map(raw_columns) -> dict[str, str]:
    mapping = dict(RENAME)
    for col in raw_columns:
        m = _Q_HEADER.match(str(col))
        if m:
            mapping[col] = f"q{int(m.group(1))}"
    return mapping


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, rename to snake_case and validate. Pure; no I/O."""
    if "Response_ID" in raw.columns and raw["Response_ID"].duplicated().any():
        raise SurveyValidationError("duplicate Response_ID: rows are not one-per-respondent")

    df = raw.drop(columns=[c for c in DROP_COLUMNS if c in raw.columns])
    mapping = _rename_map(df.columns)
    unknown = [c for c in df.columns if c not in mapping]
    if unknown:
        raise SurveyValidationError(f"unexpected columns: {unknown}")
    df = df.rename(columns=mapping)

    expected = set(RENAME.values()) | set(Q_COLUMNS)
    missing = expected - set(df.columns)
    if missing:
        raise SurveyValidationError(f"missing columns: {sorted(missing)}")

    formula_cols = [c for c in (*SCORE_COLUMNS, *SEVERITY_COLUMNS, "severity_level") if df[c].isna().all()]
    if formula_cols:
        raise SurveyValidationError(
            f"{formula_cols} are empty. These are Excel formulas and the workbook has no cached "
            "results (it was probably saved by a script, not Excel). Open it in Excel, let it "
            "recalculate, and save it again."
        )

    nulls = df.columns[df.isna().any()].tolist()
    if nulls:
        raise SurveyValidationError(f"null values in columns: {nulls}")

    for q in Q_COLUMNS:
        values = pd.to_numeric(df[q], errors="coerce")
        bad = values.isna() | (values % 1 != 0) | ~values.between(0, 3)
        if bad.any():
            raise SurveyValidationError(
                f"{q}: {int(bad.sum())} answers outside integer 0-3, "
                f"e.g. {df.loc[bad, q].iloc[0]!r}"
            )
        df[q] = values.astype(int)

    for col, options in CATEGORICAL_OPTIONS.items():
        bad = ~df[col].isin(options)
        if bad.any():
            raise SurveyValidationError(
                f"{col}: unknown options {sorted(df.loc[bad, col].unique())}"
            )

    ordered = [c for c in RENAME.values() if c not in (*SCORE_COLUMNS, *SEVERITY_COLUMNS, *LABEL_COLUMNS)]
    ordered += list(Q_COLUMNS) + list(SCORE_COLUMNS) + list(SEVERITY_COLUMNS) + list(LABEL_COLUMNS)
    return df[ordered].reset_index(drop=True)


def load_raw(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    # pandas reads the cached results of the workbook's score/severity formulas.
    return pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")


def load_survey(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
