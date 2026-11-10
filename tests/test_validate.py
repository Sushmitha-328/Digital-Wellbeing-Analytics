"""
Unit tests for src/validate.py
"""

import pytest

from src.validate import validate_inputs


# ------------------------------------------------------------------ #
#  Helper                                                             #
# ------------------------------------------------------------------ #

def _valid_inputs() -> dict:
    """A fully valid user input dictionary."""
    return {
        "age":                     21,
        "gender":                  "Female",
        "daily_screen_time_hours": 8.0,
        "social_media_hours":      3.0,
        "gaming_hours":            1.0,
        "work_study_hours":        4.0,
        "sleep_hours":             7.0,
        "notifications_per_day":   80,
        "app_opens_per_day":       60,
        "weekend_screen_time":     9.0,
        "stress_level":            "Medium",
        "academic_work_impact":    "No",
    }


# ------------------------------------------------------------------ #
#  Valid input — no errors expected                                   #
# ------------------------------------------------------------------ #

class TestValidInputs:
    def test_valid_inputs_pass(self):
        ok, errors = validate_inputs(_valid_inputs())
        assert ok is True
        assert errors == []

    def test_returns_tuple(self):
        result = validate_inputs(_valid_inputs())
        assert isinstance(result, tuple)
        assert len(result) == 2


# ------------------------------------------------------------------ #
#  Numeric range violations                                           #
# ------------------------------------------------------------------ #

class TestNumericRangeViolations:
    def test_age_too_low(self):
        data = _valid_inputs()
        data["age"] = 5
        ok, errors = validate_inputs(data)
        assert not ok
        assert any("Age" in e for e in errors)

    def test_age_too_high(self):
        data = _valid_inputs()
        data["age"] = 120
        ok, errors = validate_inputs(data)
        assert not ok

    def test_screen_time_exceeds_18h(self):
        data = _valid_inputs()
        data["daily_screen_time_hours"] = 20.0
        ok, errors = validate_inputs(data)
        assert not ok

    def test_sleep_too_low(self):
        data = _valid_inputs()
        data["sleep_hours"] = 1.0
        ok, errors = validate_inputs(data)
        assert not ok

    def test_sleep_too_high(self):
        data = _valid_inputs()
        data["sleep_hours"] = 20.0
        ok, errors = validate_inputs(data)
        assert not ok


# ------------------------------------------------------------------ #
#  Categorical violations                                             #
# ------------------------------------------------------------------ #

class TestCategoricalViolations:
    def test_invalid_gender(self):
        data = _valid_inputs()
        data["gender"] = "Robot"
        ok, errors = validate_inputs(data)
        assert not ok
        assert any("Gender" in e for e in errors)

    def test_invalid_stress_level(self):
        data = _valid_inputs()
        data["stress_level"] = "Extreme"
        ok, errors = validate_inputs(data)
        assert not ok

    def test_invalid_academic_impact(self):
        data = _valid_inputs()
        data["academic_work_impact"] = "Maybe"
        ok, errors = validate_inputs(data)
        assert not ok


# ------------------------------------------------------------------ #
#  Cross-field violations                                             #
# ------------------------------------------------------------------ #

class TestCrossFieldViolations:
    def test_total_hours_exceed_24(self):
        data = _valid_inputs()
        data["daily_screen_time_hours"] = 12.0
        data["sleep_hours"]             = 10.0
        data["work_study_hours"]        = 8.0
        ok, errors = validate_inputs(data)
        assert not ok
        assert any("24" in e for e in errors)

    def test_social_plus_gaming_exceeds_screen_time(self):
        data = _valid_inputs()
        data["daily_screen_time_hours"] = 3.0
        data["social_media_hours"]      = 2.0
        data["gaming_hours"]            = 2.0  # 4 > 3
        ok, errors = validate_inputs(data)
        assert not ok
        assert any("Inconsistent" in e for e in errors)

    def test_leisure_equals_screen_time_is_ok(self):
        """Social + gaming exactly equal to screen time should pass."""
        data = _valid_inputs()
        data["daily_screen_time_hours"] = 5.0
        data["social_media_hours"]      = 3.0
        data["gaming_hours"]            = 2.0
        ok, errors = validate_inputs(data)
        assert ok, errors

    def test_multiple_errors_returned(self):
        data = _valid_inputs()
        data["age"]   = 5
        data["gender"] = "Robot"
        ok, errors = validate_inputs(data)
        assert not ok
        assert len(errors) >= 2
