"""
Feature Engineering Module — Digital Wellbeing Analytics
=========================================================
Computes all 5 engineered behavioral features from raw inputs.

Every feature here represents a behaviorally meaningful composite.
These features capture dimensions of smartphone use that raw columns alone
cannot express — e.g., the *ratio* of screen time to sleep matters more
than either value in isolation.

NOTE: This module is the single source of truth for feature construction.
Both the training pipeline and the inference app import from here to
guarantee that training and serving always apply identical transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Column definitions (must match training data exactly)              #
# ------------------------------------------------------------------ #

RAW_FEATURES: list[str] = [
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]

ENGINEERED_FEATURES: list[str] = [
    "total_leisure_hours",
    "screen_to_sleep_ratio",
    "weekend_screen_time_increase",
    "notification_intensity",
    "productive_time_ratio",
]

ALL_MODEL_FEATURES: list[str] = RAW_FEATURES + ENGINEERED_FEATURES

CATEGORICAL_FEATURES: list[str] = ["gender", "stress_level", "academic_work_impact"]

NUMERICAL_FEATURES: list[str] = [
    f for f in ALL_MODEL_FEATURES if f not in CATEGORICAL_FEATURES
]

# Human-readable labels for the UI
FEATURE_LABELS: dict[str, str] = {
    "age": "Age",
    "gender": "Gender",
    "daily_screen_time_hours": "Daily Screen Time (hrs)",
    "social_media_hours": "Social Media Usage (hrs)",
    "gaming_hours": "Gaming Hours (hrs)",
    "work_study_hours": "Work / Study Hours (hrs)",
    "sleep_hours": "Sleep Hours (hrs)",
    "notifications_per_day": "Notifications per Day",
    "app_opens_per_day": "App Opens per Day",
    "weekend_screen_time": "Weekend Screen Time (hrs)",
    "stress_level": "Stress Level",
    "academic_work_impact": "Academic / Work Impact",
    "total_leisure_hours": "Total Leisure Hours",
    "screen_to_sleep_ratio": "Screen-to-Sleep Ratio",
    "weekend_screen_time_increase": "Weekend Screen Time Increase (hrs)",
    "notification_intensity": "Notification Intensity (notifs/open)",
    "productive_time_ratio": "Productive Time Ratio",
}


# ------------------------------------------------------------------ #
#  Core engineering function                                          #
# ------------------------------------------------------------------ #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all engineered features from raw input columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns listed in ``RAW_FEATURES``.

    Returns
    -------
    pd.DataFrame
        Original raw columns *plus* the 5 engineered features,
        in the exact column order expected by the trained model.

    Raises
    ------
    ValueError
        If any required raw column is missing.

    Engineering rationale
    ---------------------
    1. **total_leisure_hours** = social_media + gaming
       Captures total *leisure* (non-productive) screen time.
       Higher leisure hours are associated with compulsive use patterns.

    2. **screen_to_sleep_ratio** = daily_screen_time / sleep_hours
       A ratio > 1 means the user spends more waking time on screens than
       sleeping. High values indicate screen use may be encroaching on rest.
       Division-by-zero: if sleep_hours == 0, set ratio to 0 (edge case).

    3. **weekend_screen_time_increase** = weekend_screen_time - daily_screen_time
       Positive values indicate more screen use on weekends, a common
       sign of compensatory or binge use behaviour.

    4. **notification_intensity** = notifications_per_day / app_opens_per_day
       Average notifications received *per app open*. High values suggest
       the phone is "demanding attention" beyond active use.
       Division-by-zero: if app_opens == 0, set to 0.

    5. **productive_time_ratio** = work_study / (work_study + daily_screen_time)
       Fraction of combined time that is productive. Closer to 0 means
       screen time dominates over productive activity.
       Division-by-zero: if denominator == 0, set to 0.
    """
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # 1. Total leisure hours
    out["total_leisure_hours"] = out["social_media_hours"] + out["gaming_hours"]

    # 2. Screen-to-sleep ratio (guard against zero sleep)
    sleep = out["sleep_hours"].replace(0, np.nan)
    out["screen_to_sleep_ratio"] = (out["daily_screen_time_hours"] / sleep).fillna(0.0)

    # 3. Weekend screen-time increase (can be negative)
    out["weekend_screen_time_increase"] = (
        out["weekend_screen_time"] - out["daily_screen_time_hours"]
    )

    # 4. Notification intensity (guard against zero app opens)
    app_opens = out["app_opens_per_day"].replace(0, np.nan)
    out["notification_intensity"] = (
        out["notifications_per_day"] / app_opens
    ).fillna(0.0)

    # 5. Productive time ratio (guard against zero denominator)
    denominator = (out["work_study_hours"] + out["daily_screen_time_hours"]).replace(
        0, np.nan
    )
    out["productive_time_ratio"] = (out["work_study_hours"] / denominator).fillna(0.0)

    return out[ALL_MODEL_FEATURES]
