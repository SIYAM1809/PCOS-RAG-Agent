"""
LangGraph State Machine Definition (Stage B: Corrective RAG / CRAG Self-Correction Graph).

Architecture:
               ┌───────────────────────────┐
               ▼                           │
  START -> [retrieve] -> [grade_documents] ─┴─► (Relevant OR Max Retries reached) ──► [generate] ──► END
                               │
                               └─► (Irrelevant & retry_count < max_retries) ──► [transform_query] ─┘

Key Capabilities:
  1. Semantic Grading: Evaluates retrieved chunks for relevance to the user's clinical question.
  2. Query Transformation: Rewrites colloquial / lay questions into structured clinical terminology.
  3. Strict Max-Retry Protection: Hard ceiling on rewrite loops to guarantee deterministic termination.
  4. LangSmith Tracing: Live tracing of all node decisions, grades, rewrites, and generations.
"""

import os
import sys
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

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

load_dotenv()

INDEX_NAME = "PCOS_Guidelines"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


# =====================================================================
# Graph State Definition
# =====================================================================
class GraphState(TypedDict):
    question: str                  # Original user question
    search_query: str              # Active search query (may be rewritten by CRAG)
    documents: List[Any]           # Retrieved document chunks
    context: str                   # Formatted context with citation headers
    grade: str                     # Relevance grade: "yes" | "no"
    retry_count: int               # Number of query rewrites performed
    max_retries: int               # Hard cap on rewrite attempts
    answer: str                    # Final generated answer
    sources: List[Dict[str, Any]]  # List of cited source files and pages
    route_history: List[str]       # Execution trace for transparency & testing


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


# =====================================================================
# Graph Nodes
# =====================================================================
def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieves relevant guideline chunks from Weaviate using the active search_query."""
    query = state.get("search_query") or state["question"]
    route_history = list(state.get("route_history", []))
    route_history.append(f"retrieve(query='{query}')")

    print(f"\n[NODE: retrieve] Searching Weaviate with query: '{query}'...")

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
        print(f"[NODE: retrieve] Retrieved {len(docs)} chunk(s).")
        return {
            "documents": docs,
            "context": context_str,
            "sources": sources,
            "route_history": route_history,
        }
    finally:
        client.close()


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved documents contain information relevant
    to answering the original clinical question.
    """
    question = state["question"]
    context = state["context"]
    route_history = list(state.get("route_history", []))

    print(f"[NODE: grade_documents] Evaluating document relevance for: '{question}'...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.0)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval relevance grader for clinical PCOS guidelines.\n"
         "Assess whether the retrieved guideline context contains information relevant to answering the user's question.\n"
         "Answer ONLY 'yes' if the context contains relevant clinical information, or 'no' if the context is completely irrelevant, unhelpful, or off-topic.\n"
         "Output format: Output exactly one word: 'yes' or 'no'."),
        ("human", "User Question:\n{question}\n\nRetrieved Clinical Context:\n{context}\n\nIs this context relevant? (yes/no):")
    ])

    chain = grader_prompt | llm | StrOutputParser()
    raw_grade = chain.invoke({"question": question, "context": context}).strip().lower()
    grade = "yes" if "yes" in raw_grade else "no"

    route_history.append(f"grade_documents(grade='{grade}')")
    print(f"[NODE: grade_documents] Result: '{grade.upper()}' (raw LLM output: '{raw_grade}')")

    return {
        "grade": grade,
        "route_history": route_history,
    }


def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Rewrites colloquial, vague, or lay user questions into focused clinical search queries
    optimized for guideline document retrieval.
    """
    question = state["question"]
    current_retry = state.get("retry_count", 0) + 1
    route_history = list(state.get("route_history", []))

    print(f"[NODE: transform_query] Rewriting query into clinical terminology (Attempt #{current_retry})...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.1)

    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical query optimization expert for medical guideline vector search.\n"
         "The initial retrieval for the user's question yielded insufficient or low-relevance results.\n"
         "Rewrite the question into a targeted, keyword-rich clinical search query using formal medical terms (e.g., convert 'missed periods and extra hair' to 'oligomenorrhea hirsutism hyperandrogenism diagnostic evaluation PCOS').\n"
         "Output ONLY the optimized query string with no preamble or quotes."),
        ("human", "Original Question:\n{question}\n\nOptimized Clinical Search Query:")
    ])

    chain = rewriter_prompt | llm | StrOutputParser()
    new_query = chain.invoke({"question": question}).strip().replace('"', '')

    route_history.append(f"transform_query(rewritten='{new_query}', retry={current_retry})")
    print(f"[NODE: transform_query] Rewritten Query: '{new_query}'")

    return {
        "search_query": new_query,
        "retry_count": current_retry,
        "route_history": route_history,
    }


def generate(state: GraphState) -> Dict[str, Any]:
    """Generates a clinically grounded answer from retrieved context."""
    question = state["question"]
    context = state["context"]
    route_history = list(state.get("route_history", []))
    route_history.append("generate")

    print(f"[NODE: generate] Generating grounded answer with Groq ({GROQ_MODEL_NAME})...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.1)

    system_prompt = (
        "You are an expert Clinical Guideline Agent specializing in Polycystic Ovary Syndrome (PCOS).\n"
        "Your task is to answer clinical queries accurately based STRICTLY on the retrieved clinical guideline context provided below.\n\n"
        "Guidelines for your response:\n"
        "1. Ground every claim directly in the provided context. Do NOT extrapolate or hallucinate unstated medical thresholds.\n"
        "2. Explicitly cite the source document and page number for key clinical statements (e.g., [rotterdam_2003_consensus.pdf, p. 1]).\n"
        "3. If the provided context does not contain sufficient details to answer all aspects of the query, explicitly state what is missing or that the topic is not covered in the loaded guidelines.\n"
        "4. Structure your response clearly with bullet points or concise sections where appropriate.\n\n"
        "--- CLINICAL GUIDELINE CONTEXT ---\n"
        "{context}\n"
        "--- END CONTEXT ---"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    print("[NODE: generate] Generation complete.")

    return {
        "answer": answer,
        "route_history": route_history,
    }


# =====================================================================
# Conditional Edge / Routing Logic
# =====================================================================
def decide_to_generate(state: GraphState) -> Literal["generate", "transform_query"]:
    """
    Determines whether to proceed to generation or enter the query transformation loop.
    Guarantees hard-stop protection against infinite retry loops.
    """
    grade = state.get("grade", "no")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    print("\n--- [ROUTER: decide_to_generate] ---")
    print(f" • Document Grade : {grade.upper()}")
    print(f" • Retry Count    : {retry_count} / {max_retries} max")

    if grade == "yes":
        print(" -> Decision: Context is RELEVANT. Proceeding directly to 'generate'.\n")
        return "generate"

    # HARD STOP CHECK: Max retries exceeded
    if retry_count >= max_retries:
        print(f" -> Decision: Max retries ({max_retries}) reached. FORCING route to 'generate' with available context.\n")
        return "generate"

    print(f" -> Decision: Context IRRELEVANT & retries remaining. Routing to 'transform_query' (Attempt #{retry_count + 1}).\n")
    return "transform_query"


# =====================================================================
# Build and Compile LangGraph (Stage B)
# =====================================================================
def build_graph():
    """Builds the Stage B Corrective RAG (CRAG) LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # 1. Register Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)

    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        }
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    # 3. Compile Workflow
    app = workflow.compile()
    return app


# =====================================================================
# Execution & Testing Suite
# =====================================================================
def run_query(question: str, max_retries: int = 1):
    """Executes a single query through the Stage B CRAG pipeline."""
    app = build_graph()
    initial_state = {
        "question": question,
        "search_query": question,
        "documents": [],
        "context": "",
        "grade": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "answer": "",
        "sources": [],
        "route_history": [],
    }
    final_state = app.invoke(initial_state)
    return final_state


def run_crag_eval_suite():
    """
    Runs an exhaustive 7-question test suite:
      - 5 Original Baseline Questions (should pass grading on pass 1, 0 rewrites)
      - 1 Lay-Terminology Question (should trigger rewrite loop -> clinical re-retrieval -> generate)
      - 1 Out-of-Scope Question (should trigger rewrite loop -> hit max retry hard stop -> ground answer with no hallucination)
    """
    test_cases = [
        # --- Original 5 Baseline Questions ---
        {
            "id": 1,
            "category": "Baseline",
            "question": "What are the 3 core diagnostic criteria established in the Rotterdam 2003 consensus for PCOS?",
            "expected_behavior": "Pass grade on 1st attempt (0 retries)"
        },
        {
            "id": 2,
            "category": "Baseline",
            "question": "What are the specific ultrasound threshold criteria (follicle count or ovarian volume) for defining polycystic ovarian morphology?",
            "expected_behavior": "Pass grade on 1st attempt (0 retries)"
        },
        {
            "id": 3,
            "category": "Baseline",
            "question": "Why would an elevated LH to FSH ratio be observed in a patient suspected of PCOS?",
            "expected_behavior": "Pass grade on 1st attempt (0 retries)"
        },
        {
            "id": 4,
            "category": "Baseline",
            "question": "What lifestyle and behavioral interventions are recommended as the first-line management approach for PCOS?",
            "expected_behavior": "Pass grade on 1st attempt (0 retries)"
        },
        {
            "id": 5,
            "category": "Baseline",
            "question": "How should diagnostic assessment for PCOS differ between adolescents and adult women?",
            "expected_behavior": "Pass grade on 1st attempt (0 retries)"
        },

        # --- Adversarial Test Case 1: Lay Terminology ---
        {
            "id": 6,
            "category": "Lay Terminology (CRAG Rewrite Test)",
            "question": "why do I have weird skipped periods and extra dark facial hair growing?",
            "expected_behavior": "Grade 'NO' on lay query -> transform to clinical terms (oligomenorrhea/hirsutism) -> re-retrieve -> Grade 'YES' -> generate"
        },

        # --- Adversarial Test Case 2: Out of Scope ---
        {
            "id": 7,
            "category": "Out of Scope (Hard Stop & No-Hallucination Test)",
            "question": "What is the recommended dosing titration for semaglutide in pediatric differentiated thyroid carcinoma?",
            "expected_behavior": "Grade 'NO' -> transform -> re-retrieve -> Grade 'NO' -> Hit max retry hard stop (retry_count=1) -> Generate explicit 'not covered' notice"
        }
    ]

    print("=" * 80)
    print("      LANGGRAPH STAGE B (CORRECTIVE RAG / CRAG) EVALUATION SUITE")
    print("=" * 80)
    print(f"[LANGSMITH TRACING] Active: {os.getenv('LANGCHAIN_TRACING_V2')} | Project: {os.getenv('LANGCHAIN_PROJECT')}\n")

    app = build_graph()

    for item in test_cases:
        print("\n" + "=" * 80)
        print(f"[{item['id']}/7] TEST CASE ({item['category']}):")
        print(f"Question : \"{item['question']}\"")
        print(f"Expected : {item['expected_behavior']}")
        print("-" * 80)

        initial_state = {
            "question": item["question"],
            "search_query": item["question"],
            "documents": [],
            "context": "",
            "grade": "",
            "retry_count": 0,
            "max_retries": 1,
            "answer": "",
            "sources": [],
            "route_history": [],
        }

        final_state = app.invoke(initial_state)

        print("\n[CRAG EXECUTION PATHWAY]:")
        for step_idx, step in enumerate(final_state["route_history"], 1):
            print(f"  Step {step_idx}: {step}")

        print(f"\n[FINAL RETRY COUNT]: {final_state['retry_count']} (Max Allowed: {final_state['max_retries']})")
        print(f"[FINAL GRADE]      : {final_state['grade'].upper()}")

        print("\n[GENERATED CLINICAL ANSWER]:")
        print(final_state["answer"])

        print("\n[CITATIONS RETRIEVED]:")
        for s in final_state["sources"]:
            print(f"  • {s['file']} (Page {s['page']})")
        print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        run_crag_eval_suite()
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the Rotterdam criteria for diagnosing PCOS?"
        res = run_query(q)
        print("\n" + "=" * 70)
        print("[FINAL ANSWER]:\n")
        print(res["answer"])
        print("\n[EXECUTION PATH]:")
        for r in res["route_history"]:
            print(f" -> {r}")
