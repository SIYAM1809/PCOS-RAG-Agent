"""
LangGraph State Machine Definition (Stage A: Single-Agent Baseline Graph).

Architecture:
  START -> [retrieve] -> [generate] -> END

Tracks all nodes, state transitions, and LLM calls in LangSmith.
"""

import os
import sys
from typing import List, Dict, Any
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
    question: str
    documents: List[Any]
    context: str
    answer: str
    sources: List[Dict[str, Any]]


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
    """Retrieves relevant guideline chunks from Weaviate."""
    question = state["question"]
    print(f"\n[NODE: retrieve] Searching Weaviate for: '{question}'...")

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
        docs = retriever.invoke(question)

        formatted_context = []
        sources = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source_file", "Guideline")
            page = d.metadata.get("page", "?")
            formatted_context.append(f"--- [Source {i}: {src} | Page: {page}] ---\n{d.page_content.strip()}")
            sources.append({"file": src, "page": page})

        context_str = "\n\n".join(formatted_context)
        print(f"[NODE: retrieve] Retrieved {len(docs)} chunks.")
        return {
            "documents": docs,
            "context": context_str,
            "sources": sources,
        }
    finally:
        client.close()


def generate(state: GraphState) -> Dict[str, Any]:
    """Generates a clinically grounded answer from retrieved guideline context."""
    question = state["question"]
    context = state["context"]
    print(f"[NODE: generate] Invoking Groq LLM ({GROQ_MODEL_NAME})...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.1)

    system_prompt = (
        "You are an expert Clinical Guideline Agent specializing in Polycystic Ovary Syndrome (PCOS).\n"
        "Your task is to answer clinical queries accurately based STRICTLY on the retrieved clinical guideline context provided below.\n\n"
        "Guidelines for your response:\n"
        "1. Ground every claim directly in the provided context. Do NOT extrapolate or hallucinate unstated medical thresholds.\n"
        "2. Explicitly cite the source document and page number for key clinical statements (e.g., [rotterdam_2003_consensus.pdf, p. 1]).\n"
        "3. If the provided context does not contain sufficient details to answer all aspects of the query, explicitly state what is missing.\n"
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
    print("[NODE: generate] Generated response.")

    return {"answer": answer}


# =====================================================================
# Build and Compile LangGraph
# =====================================================================
def build_graph():
    """Builds the Stage A LangGraph state machine."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Add Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app


# =====================================================================
# Execution & Testing
# =====================================================================
def run_query(question: str):
    """Executes a single query through the LangGraph pipeline."""
    app = build_graph()
    initial_state = {
        "question": question,
        "documents": [],
        "context": "",
        "answer": "",
        "sources": [],
    }
    final_state = app.invoke(initial_state)
    return final_state


def run_graph_eval_suite():
    """Runs all 5 baseline questions through the LangGraph pipeline."""
    eval_set = [
        "What are the 3 core diagnostic criteria established in the Rotterdam 2003 consensus for PCOS?",
        "What are the specific ultrasound threshold criteria (follicle count or ovarian volume) for defining polycystic ovarian morphology?",
        "Why would an elevated LH to FSH ratio be observed in a patient suspected of PCOS?",
        "What lifestyle and behavioral interventions are recommended as the first-line management approach for PCOS?",
        "How should diagnostic assessment for PCOS differ between adolescents and adult women?",
    ]

    print("=" * 75)
    print("      LANGGRAPH STAGE A (SINGLE-AGENT) EVALUATION SUITE")
    print("=" * 75)
    print(f"[LANGSMITH TRACING] Active: {os.getenv('LANGCHAIN_TRACING_V2')} | Project: {os.getenv('LANGCHAIN_PROJECT')}\n")

    app = build_graph()

    for idx, q in enumerate(eval_set, 1):
        print(f"\n[{idx}/5] QUERY: {q}")
        print("-" * 75)
        state = app.invoke({"question": q, "documents": [], "context": "", "answer": "", "sources": []})

        print("\n[GRAPH GENERATED ANSWER]:")
        print(state["answer"])
        print("\n[CITATIONS RETRIEVED]:")
        for s in state["sources"]:
            print(f"  • {s['file']} (Page {s['page']})")
        print("=" * 75)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        run_graph_eval_suite()
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the Rotterdam criteria for diagnosing PCOS?"
        res = run_query(q)
        print("\n" + "=" * 70)
        print("[FINAL ANSWER]:\n")
        print(res["answer"])
        print("\n[CITATIONS]:")
        for src in res["sources"]:
            print(f" • {src['file']} (Page {src['page']})")
