"""
Supervisor Agent (agents/supervisor.py).

LLM-based semantic router that classifies incoming user queries into specialized execution routes:
  1. "guideline": Medical questions, criteria definitions, literature recommendations.
  2. "clinical": Patient tabular risk score and biomarker explanations (requires patient_data).
  3. "imaging": Ultrasound morphology analysis (requires image_path).
  4. "hybrid": Multi-modal queries combining patient biomarkers / ultrasound with guideline synthesis.

Includes safety guardrails to route to 'request_data' if required patient inputs are missing.
"""

import os
import sys
from typing import Dict, Any, Literal
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_MODEL_NAME = "openai/gpt-oss-120b"


SUPERVISOR_PROMPT = """You are an expert routing supervisor for a clinical PCOS decision support assistant.
Classify the user's query into exactly one category based on its intent:

- "guideline": General clinical questions about diagnostic criteria, consensus definitions, clinical practice guidelines, or treatment recommendations (e.g. "what are the diagnostic criteria for PCOS", "what lifestyle interventions are recommended").
- "clinical": Queries asking to evaluate, calculate, or explain a patient's own lab values, hormone ratios, symptoms, or risk score (e.g. "why was I flagged high risk given my LH:FSH ratio", "explain my lab results with LH 18 and FSH 6").
- "imaging": Queries specifically asking to analyze, interpret, or evaluate an ultrasound image or ovarian scan.
- "hybrid": Complex queries that require analyzing both patient data / ultrasound scans AND grounding the final explanation in published clinical guidelines (e.g. "why does my ultrasound and lab panel point to high risk PCOS, and what guideline steps should I take next").

User Query: {query}
Has patient lab/symptom data available in request: {has_patient_data}
Has ultrasound image available in request: {has_image}

Respond with ONLY one word: "guideline", "clinical", "imaging", or "hybrid"."""


class SupervisorAgent:
    """Semantic routing supervisor with state validation guardrails."""

    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.0)
        self.prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def classify_intent(self, query: str, has_patient_data: bool, has_image: bool) -> Literal["guideline", "clinical", "imaging", "hybrid"]:
        """Runs LLM classification on user query."""
        raw = self.chain.invoke({
            "query": query,
            "has_patient_data": str(has_patient_data),
            "has_image": str(has_image)
        }).strip().lower().replace('"', '').replace("'", "")

        # Safe fallback matching
        if "hybrid" in raw:
            return "hybrid"
        elif "clinical" in raw:
            return "clinical"
        elif "imaging" in raw or "image" in raw or "ultrasound" in raw:
            return "imaging"
        else:
            return "guideline"

    def route_query(self, state: Dict[str, Any]) -> str:
        """
        Determines the next execution node from state.
        Enforces guardrail: if patient data or image is missing for required route,
        diverts to 'request_data' clarification node.
        """
        query = state.get("query") or state.get("question", "")
        patient_data = state.get("patient_data")
        image_path = state.get("image_path")

        has_patient_data = bool(patient_data)
        has_image = bool(image_path)

        # 1. Classify intent
        route = state.get("route")
        if not route:
            route = self.classify_intent(query, has_patient_data, has_image)

        # 2. Apply Missing Data Guardrails
        if route in ("clinical", "hybrid") and not has_patient_data:
            print(f"[SUPERVISOR] Route '{route}' identified but 'patient_data' is missing -> routing to 'request_data'.")
            return "request_data"

        if route in ("imaging", "hybrid") and not has_image:
            print(f"[SUPERVISOR] Route '{route}' identified but 'image_path' is missing -> routing to 'request_data'.")
            return "request_data"

        print(f"[SUPERVISOR] Route '{route}' approved with necessary inputs.")
        return route


def test_supervisor_routing():
    """Standalone unit verification testing the Supervisor across 6 representative query types."""
    print("=" * 75)
    print("      SUPERVISOR AGENT STANDALONE ROUTING TEST")
    print("=" * 75)

    supervisor = SupervisorAgent()

    test_queries = [
        # Category 1: Guideline Queries
        {
            "id": 1,
            "query": "What are the Rotterdam criteria for diagnosing PCOS in adult women?",
            "patient_data": None,
            "image_path": None,
            "expected": "guideline"
        },
        {
            "id": 2,
            "query": "What lifestyle and dietary changes are recommended as first-line therapy?",
            "patient_data": None,
            "image_path": None,
            "expected": "guideline"
        },

        # Category 2: Clinical Lab Analysis (With vs Without Data)
        {
            "id": 3,
            "query": "Why was I flagged high risk given my LH to FSH ratio of 3.0?",
            "patient_data": {"lh": 18.0, "fsh": 6.0, "lh_fsh_ratio": 3.0, "cycle_length_days": 45},
            "image_path": None,
            "expected": "clinical"
        },
        {
            "id": 4,
            "query": "Can you analyze my risk score based on my blood test results?",
            "patient_data": None,  # Intentionally missing to test guardrail
            "image_path": None,
            "expected": "request_data"
        },

        # Category 3: Imaging / Ultrasound Queries
        {
            "id": 5,
            "query": "Please analyze this transvaginal ultrasound scan for polycystic morphology.",
            "patient_data": None,
            "image_path": "data/sample_ultrasound.png",
            "expected": "imaging"
        },

        # Category 4: Hybrid Multi-Modal Queries
        {
            "id": 6,
            "query": "Given my lab values and this ultrasound scan, explain my risk and what the guidelines recommend next.",
            "patient_data": {"lh": 15.0, "fsh": 5.0, "cycle_length_days": 50, "hirsutism_score": 7},
            "image_path": "data/sample_ultrasound.png",
            "expected": "hybrid"
        }
    ]

    for item in test_queries:
        print(f"\n[TEST {item['id']}/6]: \"{item['query']}\"")
        state = {
            "query": item["query"],
            "patient_data": item["patient_data"],
            "image_path": item["image_path"],
            "route": None
        }

        classified_intent = supervisor.classify_intent(
            item["query"],
            has_patient_data=bool(item["patient_data"]),
            has_image=bool(item["image_path"])
        )
        state["route"] = classified_intent

        routed_decision = supervisor.route_query(state)
        print(f" -> LLM Intent Class : '{classified_intent}'")
        print(f" -> Final Route Node : '{routed_decision}' (Expected: '{item['expected']}')")
        status = "PASS" if routed_decision == item["expected"] else "FAIL"
        print(f" -> Status           : [{status}]")

    print("\n" + "=" * 75)
    print(" [SUPERVISOR ROUTING TEST COMPLETED]")
    print("=" * 75)


if __name__ == "__main__":
    test_supervisor_routing()
