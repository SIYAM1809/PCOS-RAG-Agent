"""
LangGraph Multi-Agent State Machine (PCOS Care Navigator).

Architecture:
  START
    │
    ▼
[supervisor] ──► (Router: route_query)
    │
    ├──► "guideline"   ──► [guideline_crag] ───────────────────────────────────────────┐
    ├──► "clinical"    ──► [run_clinical] ─────────────────────────────────────────────┤
    ├──► "imaging"     ──► [run_imaging]  ─────────────────────────────────────────────┤
    ├──► "hybrid"      ──► [run_clinical] ──► [run_imaging] ──► [guideline_crag] ──────┤
    │                                                                                  ▼
    └──► "request_data" ──► [request_data] ──► END                             [final_synthesis] ──► END

Key Features:
  1. Semantic Intent Routing with Missing Data Guardrails.
  2. Upstream-Informed Hybrid Retrieval: Guideline query is enriched with detected clinical/imaging biomarkers.
  3. Isolated CRAG State: Zero retry/grade state leakage across runs.
  4. Discordant Phenotype Handling: Synthesizes conflicting clinical vs imaging findings per Rotterdam criteria.
  5. Fault-Tolerant Degradation: Gracefully handles corrupted/partial inputs without crashing.
  6. Rate-Limit & Token Efficiency: Optimized model tiering and automatic backoff across Groq endpoints.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import weaviate
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.clinical_agent import ClinicalAgent
from agents.imaging_agent import ImagingAgent
from agents.supervisor import SupervisorAgent

load_dotenv()

INDEX_NAME = "PCOS_Guidelines"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ROUTER_MODEL_NAME = "qwen/qwen3.8-27b"
SYNTHESIS_MODEL_NAME = "openai/gpt-oss-120b"


# =====================================================================
# Graph State Definition
# =====================================================================
class GraphState(TypedDict):
    query: str
    route: Optional[Literal["guideline", "clinical", "imaging", "hybrid", "request_data"]]
    patient_data: Optional[Dict[str, Any]]
    image_path: Optional[str]
    clinical_result: Optional[Dict[str, Any]]
    imaging_result: Optional[Dict[str, Any]]
    search_query: Optional[str]
    guideline_context: Optional[str]
    guideline_sources: Optional[List[Dict[str, Any]]]
    grade: Optional[str]
    retry_count: int
    max_retries: int
    missing_requirements: Optional[List[str]]
    final_answer: Optional[str]
    route_history: List[str]


# =====================================================================
# Resource Helper Functions
# =====================================================================
def get_weaviate_client():
    cluster_url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if not cluster_url.startswith("http://") and not cluster_url.startswith("https://"):
        cluster_url = f"https://{cluster_url}"

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )


def get_llm(model_name: str = ROUTER_MODEL_NAME, temperature: float = 0.0):
    """Creates a ChatGroq LLM instance with automatic rate-limit retry."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model=model_name,
        api_key=groq_api_key,
        temperature=temperature,
        max_retries=3,
    )


# =====================================================================
# Graph Nodes
# =====================================================================

def supervisor_node(state: GraphState) -> Dict[str, Any]:
    """Classifies user query intent and initializes route history."""
    query = state["query"]
    has_patient_data = bool(state.get("patient_data"))
    has_image = bool(state.get("image_path"))
    route_history = list(state.get("route_history", []))

    print(f"\n[NODE: supervisor] Classifying intent for: '{query}'...")
    supervisor = SupervisorAgent()
    intent = supervisor.classify_intent(query, has_patient_data, has_image)

    route_history.append(f"supervisor(intent='{intent}')")
    print(f"[NODE: supervisor] Classified intent: '{intent.upper()}' (has_data={has_patient_data}, has_image={has_image})")

    return {
        "route": intent,
        "route_history": route_history,
    }


def request_data_node(state: GraphState) -> Dict[str, Any]:
    """Generates a structured clarification request when required patient data is missing."""
    route = state.get("route")
    missing = []
    if route in ("clinical", "hybrid") and not state.get("patient_data"):
        missing.append("patient lab values / symptoms (e.g., LH, FSH, cycle length, hirsutism score, BMI)")
    if route in ("imaging", "hybrid") and not state.get("image_path"):
        missing.append("pelvic ultrasound scan image (image_path)")

    missing_text = " and ".join(missing)
    route_history = list(state.get("route_history", []))
    route_history.append("request_data")

    message = (
        f"To evaluate your query ({state['query']}), the system requires additional patient inputs: **{missing_text}**.\n\n"
        "Please provide your laboratory hormone panel or ultrasound image to proceed with clinical risk calculation."
    )
    print(f"[NODE: request_data] Requesting missing data: {missing}")

    return {
        "final_answer": message,
        "missing_requirements": missing,
        "route_history": route_history,
    }


def run_clinical_node(state: GraphState) -> Dict[str, Any]:
    """Executes tabular risk ensemble model and SHAP feature attribution."""
    patient_data = state.get("patient_data") or {}
    route_history = list(state.get("route_history", []))
    route_history.append("run_clinical")

    print("[NODE: run_clinical] Running ensemble risk assessment...")
    try:
        agent = ClinicalAgent()
        result = agent.run(patient_data)
        print(f"[NODE: run_clinical] Risk Score: {result['risk_score']} ({result['risk_label']})")
    except Exception as e:
        print(f"[NODE: run_clinical] Degraded execution: {e}")
        result = {
            "status": "degraded",
            "error": str(e),
            "risk_score": None,
            "risk_label": "Unavailable",
            "top_contributing_factors": []
        }

    return {
        "clinical_result": result,
        "route_history": route_history,
    }


def run_imaging_node(state: GraphState) -> Dict[str, Any]:
    """Executes ConvNeXt deep learning ultrasound classifier."""
    image_path = state.get("image_path")
    route_history = list(state.get("route_history", []))
    route_history.append("run_imaging")

    print(f"[NODE: run_imaging] Running ConvNeXt classifier on: '{image_path}'...")
    try:
        if not image_path:
            raise ValueError("No image path provided.")
        agent = ImagingAgent()
        result = agent.run(image_path)
        print(f"[NODE: run_imaging] Classification: {result['classification']} (prob={result['pcos_probability']})")
    except Exception as e:
        print(f"[NODE: run_imaging] Degraded execution: {e}")
        result = {
            "status": "degraded",
            "error": str(e),
            "pcos_probability": None,
            "classification": "Unavailable / Degraded Image",
            "confidence": None,
            "morphology_features": ["Image could not be processed"]
        }

    return {
        "imaging_result": result,
        "route_history": route_history,
    }


def prepare_guideline_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Constructs an informed retrieval query for the guideline agent based on
    upstream clinical / imaging findings when running in hybrid mode.
    """
    route = state.get("route")
    original_query = state["query"]
    route_history = list(state.get("route_history", []))

    # In hybrid mode, enrich the retrieval query with specific detected findings
    if route == "hybrid":
        enrichments = []
        clin = state.get("clinical_result") or {}
        for factor, _ in clin.get("top_contributing_factors", []):
            enrichments.append(str(factor))

        img = state.get("imaging_result") or {}
        if img.get("classification") == "PCOS-consistent":
            enrichments.append("polycystic ovarian morphology ultrasound threshold")

        enriched_query = f"{original_query} {' '.join(enrichments)}".strip()
        print(f"[NODE: prepare_guideline_query] Enriched hybrid query: '{enriched_query}'")
        search_query = enriched_query
    else:
        search_query = original_query

    route_history.append(f"prepare_guideline_query(query='{search_query[:40]}...')")

    # Reset CRAG-specific state variables to guarantee zero leakage between runs
    return {
        "search_query": search_query,
        "retry_count": 0,
        "grade": "",
        "route_history": route_history,
    }


def retrieve_guideline_node(state: GraphState) -> Dict[str, Any]:
    """Retrieves guideline passages from Weaviate using active search_query."""
    query = state.get("search_query") or state["query"]
    route_history = list(state.get("route_history", []))
    route_history.append(f"retrieve_guideline(query='{query[:40]}...')")

    print(f"[NODE: retrieve_guideline] Querying Weaviate for: '{query}'...")
    client = get_weaviate_client()
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text",
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)

        formatted_context = []
        sources = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source_file", "Guideline")
            page = d.metadata.get("page", "?")
            formatted_context.append(f"--- [Source {i}: {src} | Page: {page}] ---\n{d.page_content.strip()}")
            sources.append({"file": src, "page": page})

        context_str = "\n\n".join(formatted_context)
        print(f"[NODE: retrieve_guideline] Retrieved {len(docs)} chunk(s).")
        return {
            "guideline_context": context_str,
            "guideline_sources": sources,
            "route_history": route_history,
        }
    finally:
        client.close()


def grade_guideline_node(state: GraphState) -> Dict[str, Any]:
    """Evaluates guideline relevance for the active query using efficient model."""
    query = state.get("search_query") or state["query"]
    context = state.get("guideline_context") or ""
    route_history = list(state.get("route_history", []))

    print("[NODE: grade_guideline] Evaluating document relevance...")
    llm = get_llm(model_name=ROUTER_MODEL_NAME, temperature=0.0)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval relevance grader for clinical PCOS guidelines.\n"
         "Assess whether the retrieved guideline context contains information relevant to the question.\n"
         "Answer ONLY 'yes' or 'no'."),
        ("human", "Question:\n{query}\n\nContext:\n{context}\n\nIs this context relevant? (yes/no):")
    ])

    chain = grader_prompt | llm | StrOutputParser()
    try:
        raw_grade = chain.invoke({"query": query, "context": context}).strip().lower()
        grade = "yes" if "yes" in raw_grade else "no"
    except Exception as e:
        print(f"[NODE: grade_guideline] Grading warning: {e}. Defaulting to 'yes'.")
        grade = "yes"

    route_history.append(f"grade_guideline(grade='{grade}')")
    print(f"[NODE: grade_guideline] Grade: '{grade.upper()}'")

    return {
        "grade": grade,
        "route_history": route_history,
    }


def transform_guideline_query_node(state: GraphState) -> Dict[str, Any]:
    """Rewrites query to formal clinical terms upon low relevance grade."""
    query = state["query"]
    retry_count = state.get("retry_count", 0) + 1
    route_history = list(state.get("route_history", []))

    print(f"[NODE: transform_guideline_query] Rewriting query (Attempt #{retry_count})...")
    llm = get_llm(model_name=ROUTER_MODEL_NAME, temperature=0.1)

    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical query optimization expert for medical guideline retrieval.\n"
         "Rewrite the question into a targeted clinical search query using formal terminology.\n"
         "Output ONLY the query string."),
        ("human", "Original Question:\n{query}\n\nOptimized Clinical Search Query:")
    ])

    chain = rewriter_prompt | llm | StrOutputParser()
    try:
        new_query = chain.invoke({"query": query}).strip().replace('"', '')
    except Exception as e:
        print(f"[NODE: transform_guideline_query] Rewriter warning: {e}. Using original.")
        new_query = query

    route_history.append(f"transform_guideline_query(rewritten='{new_query}', retry={retry_count})")
    print(f"[NODE: transform_guideline_query] Rewritten Query: '{new_query}'")

    return {
        "search_query": new_query,
        "retry_count": retry_count,
        "route_history": route_history,
    }


def final_synthesis_node(state: GraphState) -> Dict[str, Any]:
    """
    Synthesizes clinical model outputs, imaging predictions, and guideline context
    into a coherent, evidence-grounded clinical explanation.
    Handles discordant phenotypes (e.g. normal labs + abnormal ultrasound).
    """
    route = state.get("route")
    query = state["query"]
    clinical_res = state.get("clinical_result")
    imaging_res = state.get("imaging_result")
    guideline_ctx = state.get("guideline_context") or "No external guideline passages retrieved."
    route_history = list(state.get("route_history", []))
    route_history.append("final_synthesis")

    print(f"[NODE: final_synthesis] Synthesizing response for route '{route}'...")
    llm = get_llm(model_name=SYNTHESIS_MODEL_NAME, temperature=0.1)

    system_prompt = (
        "You are an expert Clinical Decision Support Assistant specializing in Polycystic Ovary Syndrome (PCOS).\n"
        "Your role is to synthesize the provided diagnostic model outputs and clinical guideline context to answer the user's question.\n\n"
        "CRITICAL CLINICAL RULES:\n"
        "1. Never state a definitive medical diagnosis. Frame all outputs as risk indicators and decision support for clinicians.\n"
        "2. Treat model outputs (risk scores, top contributing factors, imaging classifications) as objective facts from upstream specialized agents.\n"
        "3. Ground all clinical explanations in the provided guideline context with explicit citations ([file, page]).\n"
        "4. DISCORDANT FINDINGS HANDLING: If clinical lab results and ultrasound classifications conflict (e.g. low lab risk but PCOS-consistent ultrasound, or vice-versa), explicitly highlight the discordant phenotype and explain how the Rotterdam '2-out-of-3' criteria account for such presentations.\n"
        "5. FAULT TOLERANCE: If any model output is unavailable or degraded, clearly note the missing component and summarize the remaining available evidence.\n"
        "6. If the guideline context does not cover the question, state so explicitly without hallucinating.\n\n"
        "--- CLINICAL EVIDENCE SOURCES ---\n"
        "Clinical Model Output : {clinical_result}\n"
        "Imaging Model Output  : {imaging_result}\n"
        "Guideline Context     : {guideline_context}\n"
        "--- END EVIDENCE SOURCES ---"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({
            "query": query,
            "clinical_result": str(clinical_res) if clinical_res else "N/A (Not requested)",
            "imaging_result": str(imaging_res) if imaging_res else "N/A (Not requested)",
            "guideline_context": guideline_ctx
        })
    except Exception as e:
        print(f"[NODE: final_synthesis] Synthesis with primary model failed ({e}). Falling back to fast tier...")
        fallback_llm = get_llm(model_name=ROUTER_MODEL_NAME, temperature=0.1)
        fallback_chain = prompt | fallback_llm | StrOutputParser()
        answer = fallback_chain.invoke({
            "query": query,
            "clinical_result": str(clinical_res) if clinical_res else "N/A (Not requested)",
            "imaging_result": str(imaging_res) if imaging_res else "N/A (Not requested)",
            "guideline_context": guideline_ctx
        })

    return {
        "final_answer": answer,
        "route_history": route_history,
    }


# =====================================================================
# Routing Functions
# =====================================================================

def route_query_decision(state: GraphState) -> str:
    """Routes from supervisor to the appropriate downstream agent/pipeline."""
    route = state.get("route", "guideline")
    has_patient_data = bool(state.get("patient_data"))
    has_image = bool(state.get("image_path"))

    print(f"\n[ROUTER: route_query_decision] Evaluated route: '{route}'")

    if route == "clinical":
        return "run_clinical" if has_patient_data else "request_data"
    elif route == "imaging":
        return "run_imaging" if has_image else "request_data"
    elif route == "hybrid":
        if not has_patient_data or not has_image:
            return "request_data"
        return "hybrid_pipeline"
    else:
        return "guideline_pipeline"


def decide_guideline_crag_loop(state: GraphState) -> Literal["final_synthesis", "transform_guideline_query"]:
    """CRAG self-correction router with hard-stop loop protection."""
    grade = state.get("grade", "no")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    print(f"\n[ROUTER: decide_guideline_crag_loop] Grade={grade.upper()} | Retry={retry_count}/{max_retries}")

    if grade == "yes":
        return "final_synthesis"

    if retry_count >= max_retries:
        print(f"[ROUTER] Max retries ({max_retries}) reached. Force-routing to 'final_synthesis'.")
        return "final_synthesis"

    return "transform_guideline_query"


# =====================================================================
# Build and Compile Multi-Agent Graph
# =====================================================================

def build_graph():
    """Builds and compiles the full multi-agent state machine."""
    workflow = StateGraph(GraphState)

    # 1. Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("request_data", request_data_node)
    workflow.add_node("run_clinical", run_clinical_node)
    workflow.add_node("run_imaging", run_imaging_node)
    workflow.add_node("prepare_guideline_query", prepare_guideline_query_node)
    workflow.add_node("retrieve_guideline", retrieve_guideline_node)
    workflow.add_node("grade_guideline", grade_guideline_node)
    workflow.add_node("transform_guideline_query", transform_guideline_query_node)
    workflow.add_node("final_synthesis", final_synthesis_node)

    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "supervisor")

    # Supervisor conditional router
    workflow.add_conditional_edges(
        "supervisor",
        route_query_decision,
        {
            "request_data": "request_data",
            "run_clinical": "run_clinical",
            "run_imaging": "run_imaging",
            "hybrid_pipeline": "run_clinical",
            "guideline_pipeline": "prepare_guideline_query",
        }
    )

    # Hybrid path sequential flow: Clinical -> Imaging -> Prepare Guideline Query
    def route_after_clinical(state: GraphState) -> str:
        return "run_imaging" if state.get("route") == "hybrid" else "final_synthesis"

    workflow.add_conditional_edges(
        "run_clinical",
        route_after_clinical,
        {
            "run_imaging": "run_imaging",
            "final_synthesis": "final_synthesis",
        }
    )

    def route_after_imaging(state: GraphState) -> str:
        return "prepare_guideline_query" if state.get("route") == "hybrid" else "final_synthesis"

    workflow.add_conditional_edges(
        "run_imaging",
        route_after_imaging,
        {
            "prepare_guideline_query": "prepare_guideline_query",
            "final_synthesis": "final_synthesis",
        }
    )

    # Guideline CRAG Sub-flow
    workflow.add_edge("prepare_guideline_query", "retrieve_guideline")
    workflow.add_edge("retrieve_guideline", "grade_guideline")

    workflow.add_conditional_edges(
        "grade_guideline",
        decide_guideline_crag_loop,
        {
            "final_synthesis": "final_synthesis",
            "transform_guideline_query": "transform_guideline_query",
        }
    )

    workflow.add_edge("transform_guideline_query", "retrieve_guideline")

    # Terminals
    workflow.add_edge("final_synthesis", END)
    workflow.add_edge("request_data", END)

    app = workflow.compile()
    return app


# =====================================================================
# Execution & Evaluation Runner
# =====================================================================

def run_query(query: str, patient_data: Optional[dict] = None, image_path: Optional[str] = None):
    """Executes a query through the full multi-agent graph."""
    app = build_graph()
    initial_state: GraphState = {
        "query": query,
        "route": None,
        "patient_data": patient_data,
        "image_path": image_path,
        "clinical_result": None,
        "imaging_result": None,
        "search_query": None,
        "guideline_context": None,
        "guideline_sources": [],
        "grade": None,
        "retry_count": 0,
        "max_retries": 1,
        "missing_requirements": None,
        "final_answer": None,
        "route_history": [],
    }
    return app.invoke(initial_state)


def run_full_evaluation_suite():
    """Runs an exhaustive 11-case test suite covering all routes, hybrid flows, and edge cases."""
    test_cases = [
        # --- 1. Guideline Baselines (5 cases) ---
        {
            "id": 1,
            "category": "Guideline Baseline",
            "query": "What are the 3 core diagnostic criteria established in the Rotterdam 2003 consensus for PCOS?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },
        {
            "id": 2,
            "category": "Guideline Baseline",
            "query": "What are the specific ultrasound threshold criteria (follicle count or ovarian volume) for defining polycystic ovarian morphology?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },
        {
            "id": 3,
            "category": "Guideline Baseline",
            "query": "Why would an elevated LH to FSH ratio be observed in a patient suspected of PCOS?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },
        {
            "id": 4,
            "category": "Guideline Baseline",
            "query": "What lifestyle and behavioral interventions are recommended as the first-line management approach for PCOS?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },
        {
            "id": 5,
            "category": "Guideline Baseline",
            "query": "How should diagnostic assessment for PCOS differ between adolescents and adult women?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },

        # --- 2. Guideline Adversarial (2 cases) ---
        {
            "id": 6,
            "category": "Guideline CRAG Rewrite",
            "query": "why do I have weird skipped periods and extra dark facial hair growing?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },
        {
            "id": 7,
            "category": "Guideline Out-of-Scope",
            "query": "What is the recommended dosing titration for semaglutide in pediatric differentiated thyroid carcinoma?",
            "patient_data": None,
            "image_path": None,
            "expected_route": "guideline"
        },

        # --- 3. Clinical Route (With & Without Data) ---
        {
            "id": 8,
            "category": "Clinical Route",
            "query": "Why was I flagged high risk given my LH to FSH ratio of 3.0 and cycle length of 48 days?",
            "patient_data": {"lh": 18.0, "fsh": 6.0, "lh_fsh_ratio": 3.0, "cycle_length_days": 48, "hirsutism_score": 8, "bmi": 29.5},
            "image_path": None,
            "expected_route": "clinical"
        },
        {
            "id": 9,
            "category": "Missing Data Guardrail",
            "query": "Can you analyze my blood test results and calculate my PCOS risk score?",
            "patient_data": None,  # Intentionally missing
            "image_path": None,
            "expected_route": "request_data"
        },

        # --- 4. Hybrid Route (Concordant & Discordant Findings) ---
        {
            "id": 10,
            "category": "Hybrid Concordant Route",
            "query": "Explain my risk given my elevated LH:FSH lab values and this ultrasound scan showing peripheral follicles, and outline guideline next steps.",
            "patient_data": {"lh": 18.0, "fsh": 6.0, "lh_fsh_ratio": 3.0, "cycle_length_days": 50, "hirsutism_score": 7, "bmi": 28.0},
            "image_path": "ultrasound_pcos_case_01.png",
            "expected_route": "hybrid"
        },
        {
            "id": 11,
            "category": "Hybrid Discordant Route (Low Lab Risk + High Ultrasound)",
            "query": "My blood labs were normal but my ultrasound scan showed polycystic ovaries. How do guidelines interpret this conflicting result?",
            "patient_data": {"lh": 5.0, "fsh": 5.0, "lh_fsh_ratio": 1.0, "cycle_length_days": 28, "hirsutism_score": 1, "bmi": 21.5},
            "image_path": "ultrasound_pcos_case_02.png",
            "expected_route": "hybrid"
        },
    ]

    print("=" * 80)
    print("      MULTI-AGENT LANGGRAPH FULL EVALUATION SUITE")
    print("=" * 80)
    print(f"[LANGSMITH TRACING] Active: {os.getenv('LANGCHAIN_TRACING_V2')} | Project: {os.getenv('LANGCHAIN_PROJECT')}\n")

    app = build_graph()

    for item in test_cases:
        print("\n" + "=" * 80)
        print(f"[{item['id']}/11] TEST CASE: {item['category']}")
        print(f"Query    : \"{item['query']}\"")
        print(f"Expected : {item['expected_route']}")
        print("-" * 80)

        initial_state: GraphState = {
            "query": item["query"],
            "route": None,
            "patient_data": item["patient_data"],
            "image_path": item["image_path"],
            "clinical_result": None,
            "imaging_result": None,
            "search_query": None,
            "guideline_context": None,
            "guideline_sources": [],
            "grade": None,
            "retry_count": 0,
            "max_retries": 1,
            "missing_requirements": None,
            "final_answer": None,
            "route_history": [],
        }

        res = app.invoke(initial_state)

        print("\n[EXECUTION PATHWAY]:")
        for step_idx, step in enumerate(res["route_history"], 1):
            print(f"  Step {step_idx}: {step}")

        print(f"\n[ACTUAL ROUTE]  : {res.get('route')}")
        print("\n[FINAL SYNTHESIS ANSWER]:")
        print(res["final_answer"])
        print("=" * 80)

        # 2-second rate limit safety pause between test cases
        time.sleep(2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        run_full_evaluation_suite()
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the Rotterdam criteria for diagnosing PCOS?"
        res = run_query(q)
        print("\n" + "=" * 70)
        print("[FINAL ANSWER]:\n")
        print(res["final_answer"])
