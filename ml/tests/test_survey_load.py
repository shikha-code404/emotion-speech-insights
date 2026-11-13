"""Loader validation tests on small in-memory frames (no workbook needed)."""

import pandas as pd
import pytest

from ml.src.survey.load import (
    CATEGORICAL_OPTIONS,
    DEFAULT_PATH,
    DROP_COLUMNS,
    Q_COLUMNS,
    RENAME,
    SurveyValidationError,
    clean,
    load_survey,
)


def _raw_row(**overrides):
    row = {
        "Response_ID": 1, "Timestamp": pd.Timestamp("2026-08-24"),
        "Do you agree to participate in this survey?": "Yes", "Name": "X",
        "Gender": "Prefer not to say", "Height (CM)": 170, "Weight (kg)": 65,
        "Age Group": "18 - 25", "Systolic Blood Pressure": 120,
        "Diastolic Blood Pressure": 80, "Current smoking status": "Never Smoked",
        "Current alcohol consumption status": "Former (Quite)",
        "Daily Step Count": 5000, "Sleep Duration": "7 - 8 hrs",
        "Daily Water Intake (liter)": 2,
        "Time dedicated to physical activities": "Weekly",
        "Do you tend to snack mindlessly?": "Due to Anxity",
        "Recurring physical discomforts (headaches, chest pain, etc.)": "Never",
        "Occupation": "Student",
        "Depression_Score": 0, "Anxiety_Score": 0, "Stress_Score": 0,
        "Depression_Severity": "Normal", "Anxiety_Severity": "Normal",
        "Stress_Severity": "Normal", "Emotional_State": "Normal", "Severity_Level": "Normal",
    }
    for i in range(1, 22):
        row[f"Q{i}. item text {i}"] = 0
    row.update(overrides)
    return row


def test_clean_renames_and_drops():
    df = clean(pd.DataFrame([_raw_row()]))
    assert set(df.columns) == set(RENAME.values()) | set(Q_COLUMNS)
    for col in DROP_COLUMNS:
        assert col not in df.columns
    assert df.loc[0, "alcohol_status"] == "Former (Quite)"       # exact spelling kept
    assert df.loc[0, "gender"] == "Prefer not to say"             # kept as its own category


@pytest.mark.parametrize("value", [4, -1, 1.5, "two"])
def test_out_of_range_answer_rejected(value):
    with pytest.raises(SurveyValidationError, match="q7"):
        clean(pd.DataFrame([_raw_row(**{"Q7. item text 7": value})]))


def test_null_rejected():
    with pytest.raises(SurveyValidationError, match="null"):
        clean(pd.DataFrame([_raw_row(**{"Sleep Duration": None})]))


def test_missing_formula_cache_gets_clear_error():
    """A workbook re-saved without cached formula results reads as all-NaN scores."""
    rows = [_raw_row(Response_ID=i, Depression_Score=None, Depression_Severity=None) for i in (1, 2)]
    with pytest.raises(SurveyValidationError, match="Excel formulas"):
        clean(pd.DataFrame(rows))


def test_unknown_option_rejected():
    with pytest.raises(SurveyValidationError, match="alcohol_status"):
        clean(pd.DataFrame([_raw_row(**{"Current alcohol consumption status": "Former (Quit)"})]))


def test_duplicate_respondent_rejected():
    with pytest.raises(SurveyValidationError, match="Response_ID"):
        clean(pd.DataFrame([_raw_row(), _raw_row()]))


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="survey workbook not present")
def test_real_workbook_loads_clean():
    df = load_survey()
    assert df.shape == (1200, 44)
    for col, options in CATEGORICAL_OPTIONS.items():
        assert set(df[col]) == set(options), col   # every allowed option is observed
