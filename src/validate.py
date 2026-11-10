"""
Input Validation Module — Digital Wellbeing Analytics
======================================================
Validates user inputs for biological and behavioural plausibility before
passing them to the model.  Invalid inputs can cause the model to produce
misleading predictions silently — this module catches those cases early and
returns human-readable error messages.
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------ #
#  Per-field range constraints                                        #
# ------------------------------------------------------------------ #

_FIELD_RANGES: dict[str, tuple[float, float]] = {
    "age":                      (10,   80),
    "daily_screen_time_hours":  (0.0,  18.0),  # Max 18 h — some sleep required
    "social_media_hours":       (0.0,  16.0),
    "gaming_hours":             (0.0,  16.0),
    "work_study_hours":         (0.0,  16.0),
    "sleep_hours":              (3.0,  14.0),  # Clinically feasible range
    "notifications_per_day":    (0,    3000),
    "app_opens_per_day":        (1,    2000),   # Min 1 to avoid division by zero
    "weekend_screen_time":      (0.0,  18.0),
}

_VALID_CATEGORICALS: dict[str, list[str]] = {
    "gender":               ["Male", "Female", "Other"],
    "stress_level":         ["Low", "Medium", "High"],
    "academic_work_impact": ["Yes", "No"],
}

_FIELD_LABELS: dict[str, str] = {
    "age":                      "Age",
    "daily_screen_time_hours":  "Daily Screen Time",
    "social_media_hours":       "Social Media Usage",
    "gaming_hours":             "Gaming Hours",
    "work_study_hours":         "Work / Study Hours",
    "sleep_hours":              "Sleep Hours",
    "notifications_per_day":    "Notifications per Day",
    "app_opens_per_day":        "App Opens per Day",
    "weekend_screen_time":      "Weekend Screen Time",
    "gender":                   "Gender",
    "stress_level":             "Stress Level",
    "academic_work_impact":     "Academic / Work Impact",
}


def validate_inputs(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single user's raw inputs.

    Parameters
    ----------
    data : dict
        Keys are raw feature names; values are the user-supplied values.

    Returns
    -------
    (is_valid, errors)
        ``is_valid`` is ``True`` only when ``errors`` is empty.
        ``errors`` is a list of human-readable error strings.
    """
    errors: list[str] = []

    # ── 1. Numeric range checks ─────────────────────────────────── #
    for field, (lo, hi) in _FIELD_RANGES.items():
        if field not in data:
            continue
        val = data[field]
        label = _FIELD_LABELS.get(field, field)
        if not (lo <= val <= hi):
            errors.append(
                f"**{label}**: {val} is outside the valid range "
                f"[{lo}, {hi}]."
            )

    # ── 2. Categorical value checks ─────────────────────────────── #
    for field, valid_values in _VALID_CATEGORICALS.items():
        if field not in data:
            continue
        val = data[field]
        label = _FIELD_LABELS.get(field, field)
        if val not in valid_values:
            errors.append(
                f"**{label}**: '{val}' is not a valid value. "
                f"Must be one of: {valid_values}."
            )

    # ── 3. Cross-field plausibility checks ──────────────────────── #
    screen = data.get("daily_screen_time_hours", 0)
    sleep  = data.get("sleep_hours",             0)
    work   = data.get("work_study_hours",        0)
    social = data.get("social_media_hours",      0)
    gaming = data.get("gaming_hours",            0)

    # Total day hours (simplified — excludes eating, commuting, etc.)
    total = screen + sleep + work
    if total > 24:
        errors.append(
            f"**Impossible schedule**: Screen ({screen}h) + Sleep ({sleep}h) "
            f"+ Work/Study ({work}h) = {total:.1f}h, which exceeds 24 hours."
        )

    # Social + gaming cannot exceed total screen time
    leisure = social + gaming
    if leisure > screen + 0.5:          # 0.5 h tolerance for rounding
        errors.append(
            f"**Inconsistent hours**: Social media ({social}h) + Gaming "
            f"({gaming}h) = {leisure}h cannot exceed total screen time "
            f"({screen}h)."
        )

    return len(errors) == 0, errors


def get_field_ranges() -> dict[str, tuple[float, float]]:
    """Return the per-field numeric range constraints (read-only copy)."""
    return dict(_FIELD_RANGES)
