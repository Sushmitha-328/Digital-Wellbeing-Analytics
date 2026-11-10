"""
Inference / Prediction Service — Digital Wellbeing Analytics
=============================================================
Wraps the trained sklearn Pipeline to produce:
  - binary prediction
  - risk probability
  - risk category label
  - top behavioural risk factors (based on RF feature importances)
  - personalised digital-wellbeing suggestions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Lazy imports (avoid hard failures at startup when optional libs     #
#  are missing)                                                        #
# ------------------------------------------------------------------ #
try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise ImportError("joblib is required: pip install joblib") from exc

from src.features import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    engineer_features,
)

# ------------------------------------------------------------------ #
#  Risk thresholds (documented here, not in the UI layer)             #
# ------------------------------------------------------------------ #
#  Thresholds were chosen to reflect three practically meaningful      #
#  segments of the probability distribution:                          #
#    LOW       p < 0.35  — user's pattern resembles non-addicted users #
#    MODERATE  0.35 ≤ p < 0.65  — ambiguous / borderline behaviour    #
#    HIGH      p ≥ 0.65  — user's pattern strongly resembles addicted  #
#                           users in the training data                 #
#  These thresholds are intentionally conservative: we prefer to flag  #
#  borderline cases as MODERATE rather than LOW.                       #
# ------------------------------------------------------------------ #
RISK_LOW_THRESHOLD      = 0.35
RISK_HIGH_THRESHOLD     = 0.65


# ------------------------------------------------------------------ #
#  Suggestion rules                                                   #
# ------------------------------------------------------------------ #
_SUGGESTIONS: list[dict] = [
    {
        "condition": lambda d: d.get("daily_screen_time_hours", 0) > 6,
        "text": (
            "📱 Consider setting a **daily screen time limit** (e.g. 5–6 hours). "
            "Use built-in Digital Wellbeing / Screen Time tools on your device."
        ),
    },
    {
        "condition": lambda d: d.get("social_media_hours", 0) > 3,
        "text": (
            "📲 Try **scheduled social media windows** instead of browsing freely. "
            "Many people find 30-minute focused sessions reduce compulsive checking."
        ),
    },
    {
        "condition": lambda d: d.get("gaming_hours", 0) > 3,
        "text": (
            "🎮 Consider **session timers** for gaming. Aim for natural break points "
            "rather than open-ended sessions."
        ),
    },
    {
        "condition": lambda d: d.get("sleep_hours", 8) < 7,
        "text": (
            "😴 You may be getting less sleep than recommended (7–9 hours for adults). "
            "Try a **phone-free wind-down routine** at least 30 minutes before bed."
        ),
    },
    {
        "condition": lambda d: d.get("notifications_per_day", 0) > 100,
        "text": (
            "🔕 With many daily notifications, consider **turning off non-essential "
            "app alerts**. Batch checking email/messages 2–3 times per day can "
            "significantly reduce reactive phone use."
        ),
    },
    {
        "condition": lambda d: (
            d.get("weekend_screen_time", 0) - d.get("daily_screen_time_hours", 0) > 2
        ),
        "text": (
            "📅 Your weekend screen time is notably higher than weekdays — a pattern "
            "sometimes linked to compensatory use. Try **scheduling offline activities** "
            "(walks, hobbies, social time) on weekends."
        ),
    },
    {
        "condition": lambda d: d.get("stress_level") == "High",
        "text": (
            "🧘 High stress can increase compulsive phone checking as a coping "
            "mechanism. Explore **stress-management techniques** such as brief "
            "mindfulness exercises or structured breaks."
        ),
    },
    {
        "condition": lambda d: d.get("academic_work_impact") == "Yes",
        "text": (
            "📚 If smartphone use is affecting your academic or work performance, "
            "try **app blocking tools** during study/work hours "
            "(e.g. Forest, Freedom, Focus@Will)."
        ),
    },
]

_LOW_RISK_SUGGESTIONS: list[str] = [
    "✅ Your current smartphone usage patterns look **healthy**.",
    "🔄 Maintain regular offline activities and face-to-face social interactions.",
    "🌙 Continue prioritising quality sleep — it's a strong protective factor.",
]


# ------------------------------------------------------------------ #
#  Result dataclass                                                   #
# ------------------------------------------------------------------ #

@dataclass
class PredictionResult:
    """Structured output from the prediction service."""

    probability:      float          # P(addicted)  in [0, 1]
    risk_category:    str            # 'LOW' | 'MODERATE' | 'HIGH'
    top_risk_factors: list[dict]     # [{'feature': str, 'importance': float, 'value': Any}, ...]
    suggestions:      list[str]      # Personalised advice strings
    engineered_df:    pd.DataFrame   # Full feature row fed to the model


# ------------------------------------------------------------------ #
#  Prediction service                                                 #
# ------------------------------------------------------------------ #

class DigitalWellbeingPredictor:
    """Wraps a trained sklearn Pipeline for inference.

    Parameters
    ----------
    model_path : str | Path
        Path to the ``joblib``-serialised Pipeline.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._pipeline   = joblib.load(self._model_path)
        self._classifier = self._pipeline.named_steps["classifier"]
        self._preprocessor = self._pipeline.named_steps["preprocessor"]
        self._feature_importances = self._classifier.feature_importances_
        self._transformed_feature_names = self._get_transformed_feature_names()

    # ── Private helpers ──────────────────────────────────────────── #

    def _get_transformed_feature_names(self) -> list[str]:
        """Return feature names *after* preprocessing (OHE + Scaler)."""
        try:
            cat_transformer = self._preprocessor.named_transformers_["categorical"]
            cat_names = list(
                cat_transformer.get_feature_names_out(CATEGORICAL_FEATURES)
            )
        except AttributeError:
            cat_names = []

        num_names = [
            c for c in ALL_MODEL_FEATURES if c not in CATEGORICAL_FEATURES
        ]
        return cat_names + num_names

    def _get_top_factors(
        self, engineered_df: pd.DataFrame, n: int = 5
    ) -> list[dict]:
        """
        Return the top-n most important raw/engineered features for this
        prediction, based on global RF feature importances mapped back to
        original feature names (categorical OHE columns are aggregated).
        """
        # Aggregate OHE importances back to original categorical columns
        importance_map: dict[str, float] = {}

        for name, imp in zip(self._transformed_feature_names, self._feature_importances):
            # OHE names look like "x0_Male", "x1_High", etc.
            # or in newer sklearn: "gender_Male", "stress_level_High", etc.
            original = name
            for cat in CATEGORICAL_FEATURES:
                if name.startswith(cat + "_") or name.startswith(f"x{CATEGORICAL_FEATURES.index(cat)}_"):
                    original = cat
                    break
            importance_map[original] = importance_map.get(original, 0.0) + imp

        # Sort by importance descending
        sorted_features = sorted(
            importance_map.items(), key=lambda x: x[1], reverse=True
        )[:n]

        result = []
        for feat, imp in sorted_features:
            val = engineered_df.iloc[0].get(feat, "N/A")
            result.append(
                {
                    "feature":    feat,
                    "label":      FEATURE_LABELS.get(feat, feat),
                    "importance": round(imp, 4),
                    "value":      val,
                }
            )
        return result

    @staticmethod
    def _get_suggestions(raw_inputs: dict[str, Any]) -> list[str]:
        triggered = [s["text"] for s in _SUGGESTIONS if s["condition"](raw_inputs)]
        if not triggered:
            return list(_LOW_RISK_SUGGESTIONS)
        return triggered

    @staticmethod
    def _risk_category(prob: float) -> str:
        if prob < RISK_LOW_THRESHOLD:
            return "LOW"
        if prob < RISK_HIGH_THRESHOLD:
            return "MODERATE"
        return "HIGH"

    # ── Public API ───────────────────────────────────────────────── #

    def predict(self, raw_inputs: dict[str, Any]) -> PredictionResult:
        """Run the full prediction pipeline for a single user.

        Parameters
        ----------
        raw_inputs : dict
            Keys are raw feature names (see ``src.features.RAW_FEATURES``).

        Returns
        -------
        PredictionResult
        """
        raw_df       = pd.DataFrame([raw_inputs])
        engineered   = engineer_features(raw_df)

        prob         = float(self._pipeline.predict_proba(engineered)[0, 1])
        category     = self._risk_category(prob)
        top_factors  = self._get_top_factors(engineered)
        suggestions  = self._get_suggestions(raw_inputs)

        return PredictionResult(
            probability      = prob,
            risk_category    = category,
            top_risk_factors = top_factors,
            suggestions      = suggestions,
            engineered_df    = engineered,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on a DataFrame of raw inputs.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all columns in ``src.features.RAW_FEATURES``.

        Returns
        -------
        pd.DataFrame
            Original columns plus:
            - ``addicted_probability``
            - ``risk_category``
        """
        engineered = engineer_features(df)
        probs      = self._pipeline.predict_proba(engineered)[:, 1]
        out        = df.copy()
        out["addicted_probability"] = np.round(probs, 4)
        out["risk_category"]        = [self._risk_category(p) for p in probs]
        return out
