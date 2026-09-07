"""
Clinical Agent Tool Wrapper (agents/clinical_agent.py).

Wraps the tabular ensemble risk prediction model (XGBoost/LightGBM/CatBoost/Voting)
and extracts SHAP feature attribution values for top contributing risk factors.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ClinicalAgent:
    """Wraps tabular ensemble model and generates risk scores with SHAP explanations."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Look for common ensemble filenames in models/
            candidates = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.pkl"))
            if candidates:
                model_path = str(candidates[0])
            else:
                model_path = str(MODELS_DIR / "pcos_ensemble_model.joblib")

        self.model_path = model_path
        self.model = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        """Loads model and initializes SHAP explainer if artifact exists."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                import shap
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception:
                    self.explainer = shap.Explainer(self.model)
                print(f"[ClinicalAgent] Loaded ensemble model from: {self.model_path}")
            except Exception as e:
                print(f"[ClinicalAgent] Could not load model from {self.model_path}: {e}")
                self.model = None
        else:
            print(f"[ClinicalAgent] Model file not found at {self.model_path}. Running with clinical rules fallback.")

    def _prepare_features(self, patient_data: dict) -> Tuple[np.ndarray, List[str]]:
        """Extracts and normalizes features from patient data dictionary."""
        feature_names = [
            "lh_fsh_ratio",
            "bmi",
            "cycle_length_days",
            "free_testosterone",
            "amh_level",
            "fasting_glucose",
            "hirsutism_score"
        ]

        # Extract features with clinical default handling
        lh = float(patient_data.get("lh", 6.0))
        fsh = float(patient_data.get("fsh", 6.0))
        lh_fsh = patient_data.get("lh_fsh_ratio", lh / fsh if fsh > 0 else 1.0)

        bmi = float(patient_data.get("bmi", 24.0))
        cycle = float(patient_data.get("cycle_length_days", 28.0))
        testo = float(patient_data.get("free_testosterone", 1.5))
        amh = float(patient_data.get("amh_level", 2.0))
        glucose = float(patient_data.get("fasting_glucose", 90.0))
        mfg = float(patient_data.get("hirsutism_score", 2.0))

        values = [lh_fsh, bmi, cycle, testo, amh, glucose, mfg]
        return np.array([values]), feature_names

    def _top_shap_factors(self, shap_vals, feature_names: List[str], n: int = 3) -> List[Tuple[str, float]]:
        """Extracts top positive contributors to risk."""
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif hasattr(shap_vals, "values"):
            vals = shap_vals.values[0]
        else:
            vals = np.array(shap_vals)[0]

        impacts = [(feature_names[i], float(vals[i])) for i in range(len(feature_names))]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        return impacts[:n]

    def run(self, patient_data: dict) -> dict:
        """Runs inference and returns risk score and SHAP attribution factors."""
        features, feat_names = self._prepare_features(patient_data)

        if self.model is not None:
            try:
                risk_score = float(self.model.predict_proba(features)[0][1])
                shap_values = self.explainer.shap_values(features)
                top_factors = self._top_shap_factors(shap_values, feat_names, n=3)
            except Exception as e:
                print(f"[ClinicalAgent] Prediction error: {e}. Falling back to baseline scoring.")
                risk_score, top_factors = self._rule_based_fallback(patient_data)
        else:
            risk_score, top_factors = self._rule_based_fallback(patient_data)

        risk_label = "High Risk" if risk_score >= 0.5 else "Low Risk"

        return {
            "risk_score": round(risk_score, 4),
            "risk_label": risk_label,
            "top_contributing_factors": top_factors,
            "patient_summary": {
                "lh_fsh_ratio": patient_data.get("lh_fsh_ratio", round(patient_data.get("lh", 6) / max(patient_data.get("fsh", 6), 0.1), 2)),
                "cycle_length_days": patient_data.get("cycle_length_days", 28),
                "hirsutism_score": patient_data.get("hirsutism_score", 0),
                "bmi": patient_data.get("bmi", 24)
            }
        }

    def _rule_based_fallback(self, patient_data: dict) -> Tuple[float, List[Tuple[str, float]]]:
        """Clinically calibrated fallback scoring when model weights are loading."""
        lh_fsh = patient_data.get("lh_fsh_ratio", patient_data.get("lh", 6.0) / max(patient_data.get("fsh", 6.0), 0.1))
        cycle = patient_data.get("cycle_length_days", 28)
        mfg = patient_data.get("hirsutism_score", 2)
        bmi = patient_data.get("bmi", 24)

        score = 0.15
        factors = []

        if lh_fsh >= 2.0:
            score += 0.35
            factors.append(("LH:FSH ratio >= 2.0 (neuroendocrine marker)", 0.35))
        if cycle >= 35 or cycle <= 21:
            score += 0.30
            factors.append((f"Irregular cycle length ({cycle} days)", 0.30))
        if mfg >= 6:
            score += 0.20
            factors.append((f"Modified Ferriman-Gallwey score ({mfg})", 0.20))
        if bmi >= 28:
            score += 0.10
            factors.append((f"Elevated BMI ({bmi})", 0.10))

        score = min(score, 0.95)
        if not factors:
            factors.append(("Normal hormonal and clinical baseline", 0.05))

        return score, factors[:3]


def test_clinical_agent():
    """Standalone unit verification on sample patient profiles."""
    print("=" * 70)
    print("      CLINICAL AGENT STANDALONE TEST")
    print("=" * 70)

    agent = ClinicalAgent()

    test_patients = [
        {
            "name": "Patient A (High Risk Profile)",
            "data": {"lh": 18.0, "fsh": 6.0, "lh_fsh_ratio": 3.0, "cycle_length_days": 48, "hirsutism_score": 8, "bmi": 29.5}
        },
        {
            "name": "Patient B (Normal / Low Risk Profile)",
            "data": {"lh": 5.2, "fsh": 5.0, "lh_fsh_ratio": 1.04, "cycle_length_days": 28, "hirsutism_score": 2, "bmi": 22.0}
        },
        {
            "name": "Patient C (Borderline Oligomenorrhea Profile)",
            "data": {"lh": 10.5, "fsh": 5.0, "lh_fsh_ratio": 2.1, "cycle_length_days": 36, "hirsutism_score": 4, "bmi": 25.0}
        }
    ]

    for p in test_patients:
        print(f"\n[TEST]: {p['name']}")
        result = agent.run(p["data"])
        print(f" -> Risk Score : {result['risk_score']} ({result['risk_label']})")
        print(" -> Top Factors:")
        for factor, weight in result["top_contributing_factors"]:
            print(f"    • {factor}: {weight}")

    print("\n" + "=" * 70)
    print(" [CLINICAL AGENT TEST COMPLETED]")
    print("=" * 70)


if __name__ == "__main__":
    test_clinical_agent()
