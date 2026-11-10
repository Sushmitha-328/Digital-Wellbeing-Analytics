"""
Unit tests for src/features.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ALL_MODEL_FEATURES,
    ENGINEERED_FEATURES,
    RAW_FEATURES,
    engineer_features,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #

def _base_row() -> dict:
    """A typical, valid single-user row."""
    return {
        "age":                     21,
        "gender":                  "Female",
        "daily_screen_time_hours": 8.0,
        "social_media_hours":      4.0,
        "gaming_hours":            2.0,
        "work_study_hours":        4.0,
        "sleep_hours":             7.0,
        "notifications_per_day":   150,
        "app_opens_per_day":       100,
        "weekend_screen_time":     10.0,
        "stress_level":            "High",
        "academic_work_impact":    "Yes",
    }


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    return pd.DataFrame([_base_row()])


@pytest.fixture
def multi_row_df() -> pd.DataFrame:
    rows = [_base_row() for _ in range(5)]
    rows[1]["sleep_hours"] = 6.0
    rows[2]["gaming_hours"] = 0.0
    rows[3]["daily_screen_time_hours"] = 4.0
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Output shape and columns                                           #
# ------------------------------------------------------------------ #

class TestOutputStructure:
    def test_output_has_all_model_features(self, single_row_df):
        out = engineer_features(single_row_df)
        assert list(out.columns) == ALL_MODEL_FEATURES

    def test_raw_features_preserved(self, single_row_df):
        out = engineer_features(single_row_df)
        for col in RAW_FEATURES:
            assert col in out.columns

    def test_engineered_features_added(self, single_row_df):
        out = engineer_features(single_row_df)
        for col in ENGINEERED_FEATURES:
            assert col in out.columns

    def test_row_count_preserved(self, multi_row_df):
        out = engineer_features(multi_row_df)
        assert len(out) == len(multi_row_df)


# ------------------------------------------------------------------ #
#  Correctness of each engineered feature                             #
# ------------------------------------------------------------------ #

class TestFeatureValues:
    def test_total_leisure_hours(self, single_row_df):
        out = engineer_features(single_row_df)
        expected = 4.0 + 2.0  # social_media + gaming
        assert out["total_leisure_hours"].iloc[0] == pytest.approx(expected)

    def test_screen_to_sleep_ratio(self, single_row_df):
        out = engineer_features(single_row_df)
        expected = 8.0 / 7.0
        assert out["screen_to_sleep_ratio"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_weekend_screen_time_increase(self, single_row_df):
        out = engineer_features(single_row_df)
        expected = 10.0 - 8.0  # weekend - weekday
        assert out["weekend_screen_time_increase"].iloc[0] == pytest.approx(expected)

    def test_notification_intensity(self, single_row_df):
        out = engineer_features(single_row_df)
        expected = 150 / 100  # notifications / app_opens
        assert out["notification_intensity"].iloc[0] == pytest.approx(expected)

    def test_productive_time_ratio(self, single_row_df):
        out = engineer_features(single_row_df)
        expected = 4.0 / (4.0 + 8.0)
        assert out["productive_time_ratio"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_negative_weekend_increase_allowed(self):
        """Weekend screen time can be lower than weekday (negative increase)."""
        row = _base_row()
        row["weekend_screen_time"] = 5.0
        row["daily_screen_time_hours"] = 8.0
        out = engineer_features(pd.DataFrame([row]))
        assert out["weekend_screen_time_increase"].iloc[0] == pytest.approx(-3.0)


# ------------------------------------------------------------------ #
#  Edge cases — division by zero                                      #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_zero_sleep_no_division_error(self):
        row = _base_row()
        row["sleep_hours"] = 0.0
        out = engineer_features(pd.DataFrame([row]))
        assert out["screen_to_sleep_ratio"].iloc[0] == 0.0
        assert not np.isnan(out["screen_to_sleep_ratio"].iloc[0])

    def test_zero_app_opens_no_division_error(self):
        row = _base_row()
        row["app_opens_per_day"] = 0
        out = engineer_features(pd.DataFrame([row]))
        assert out["notification_intensity"].iloc[0] == 0.0

    def test_zero_screen_and_work_no_division_error(self):
        row = _base_row()
        row["daily_screen_time_hours"] = 0.0
        row["work_study_hours"] = 0.0
        out = engineer_features(pd.DataFrame([row]))
        assert out["productive_time_ratio"].iloc[0] == 0.0

    def test_missing_column_raises_value_error(self):
        df = pd.DataFrame([{"age": 21}])  # many columns missing
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer_features(df)
