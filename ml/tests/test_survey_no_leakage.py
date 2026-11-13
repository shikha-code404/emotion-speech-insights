"""Leakage guard: no DASS-21 item, score, severity or label column may reach
the lifestyle model (Emotional_State is a deterministic function of q1..q21)."""

import joblib
import pandas as pd
import pytest

from ml.src.survey.load import (
    DEFAULT_PATH,
    LABEL_COLUMNS,
    Q_COLUMNS,
    SCORE_COLUMNS,
    SEVERITY_COLUMNS,
    load_survey,
)
from ml.src.survey.train_lifestyle import (
    ARTIFACT_DIR,
    FORBIDDEN_COLUMNS,
    LIFESTYLE_FEATURES,
    TARGET,
    assert_no_leakage,
    make_X,
    model_specs,
)

ALL_FORBIDDEN = set(Q_COLUMNS) | set(SCORE_COLUMNS) | set(SEVERITY_COLUMNS) | set(LABEL_COLUMNS)
needs_data = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="survey workbook not present")


def test_forbidden_set_covers_every_item_score_severity_and_label():
    assert ALL_FORBIDDEN <= FORBIDDEN_COLUMNS
    assert len([c for c in FORBIDDEN_COLUMNS if c.startswith("q")]) == 21


def test_feature_list_is_disjoint_from_forbidden():
    assert set(LIFESTYLE_FEATURES).isdisjoint(ALL_FORBIDDEN)
    assert TARGET not in LIFESTYLE_FEATURES


@pytest.mark.parametrize("col", ["q3", "q21", "stress_score", "anxiety_severity",
                                 "emotional_state", "severity_level", "Depression_Score"])
def test_assert_no_leakage_rejects(col):
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(["bmi", col])


@needs_data
def test_make_X_returns_exactly_lifestyle_features():
    df = load_survey()
    X = make_X(df)
    assert list(X.columns) == list(LIFESTYLE_FEATURES)
    assert set(X.columns).isdisjoint(ALL_FORBIDDEN)


@needs_data
def test_make_X_ignores_dass_columns():
    """Scrambling every DASS column must not change the feature frame."""
    df = load_survey()
    scrambled = df.copy()
    for col in ALL_FORBIDDEN:
        scrambled[col] = df[col].sample(frac=1, random_state=0).to_numpy()
    pd.testing.assert_frame_equal(make_X(df), make_X(scrambled))


@needs_data
@pytest.mark.parametrize("name", list(model_specs()))
def test_fitted_pipelines_only_see_lifestyle_features(name):
    df = load_survey()
    pipeline, _ = model_specs()[name]
    pipeline.fit(make_X(df), df[TARGET])
    assert list(pipeline.feature_names_in_) == list(LIFESTYLE_FEATURES)
    out = pipeline.named_steps["prep"].get_feature_names_out()
    assert not any(f.split("__", 1)[1].startswith(tuple(ALL_FORBIDDEN)) for f in out)


@pytest.mark.skipif(not (ARTIFACT_DIR / "lifestyle_model.joblib").exists(), reason="artifact not exported")
def test_exported_model_has_no_leaked_inputs():
    model = joblib.load(ARTIFACT_DIR / "lifestyle_model.joblib")
    assert list(model.feature_names_in_) == list(LIFESTYLE_FEATURES)
    assert_no_leakage(model.feature_names_in_)
