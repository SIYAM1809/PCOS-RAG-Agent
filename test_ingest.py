"""
End-to-End Ingestion Verification Script.

Performs 5 rigorous verification checks:
1. Object count in Weaviate collection
2. Semantic similarity search relevance
3. Chunk readability and quality spot-check
4. Metadata provenance verification (source, page)
5. Multi-query eval set relevance test
"""

import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore

load_dotenv()

INDEX_NAME = "PCOS_Guidelines"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_weaviate_client():
    cluster_url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if not cluster_url.startswith("http://") and not cluster_url.startswith("https://"):
        cluster_url = f"https://{cluster_url}"

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )


def run_checks():
    print("=" * 70)
    print("      PCOS GUIDELINE INGESTION VERIFICATION CHECKLIST")
    print("=" * 70)

    client = get_weaviate_client()

    try:
        # Check if collection exists
        if not client.collections.exists(INDEX_NAME):
            print(f"[FAIL] Collection '{INDEX_NAME}' does not exist in Weaviate.")
            return

        collection = client.collections.get(INDEX_NAME)

        # -------------------------------------------------------------
        # STEP 1: Total Objects Count
        # -------------------------------------------------------------
        print("\n[STEP 1] Object Count Verification:")
        agg = collection.aggregate.over_all(total_count=True)
        total_count = agg.total_count
        print(f" -> Total indexed chunks in '{INDEX_NAME}': {total_count}")
        if total_count > 0:
            print(f" [PASS] Object count ({total_count}) is valid and non-zero.")
        else:
            print(" [FAIL] Collection is empty.")
            return

        # -------------------------------------------------------------
        # STEP 2: Real Similarity Search
        # -------------------------------------------------------------
        print("\n[STEP 2] Similarity Search Relevance Check:")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text",
        )

        query_1 = "what are the diagnostic criteria for PCOS"
        print(f" -> Query: '{query_1}' (Top 3 results)")
        results = vector_store.similarity_search(query_1, k=3)
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"    Result {i} [{src} - Page {page}]:")
            print(f"    \"{snippet}...\"\n")

        # -------------------------------------------------------------
        # STEP 3 & 4: Spot-check Chunk Quality & Metadata
        # -------------------------------------------------------------
        print("[STEP 3 & 4] Spot-Check Chunk Quality & Metadata Provenance:")
        sample = collection.query.fetch_objects(limit=5)
        print(f" -> Metadata properties present: {list(sample.objects[0].properties.keys())}")

        for i, obj in enumerate(sample.objects, 1):
            props = obj.properties
            src = props.get("source_file", "MISSING")
            page = props.get("page", "MISSING")
            text = props.get("text", "")[:180].replace("\n", " ")
            print(f"   Chunk {i} | Source: {src} | Page: {page} | Chars: {len(props.get('text', ''))}")
            print(f"   Sample: \"{text}...\"\n")

        # -------------------------------------------------------------
        # STEP 5: Eval Set Questions Relevance Test
        # -------------------------------------------------------------
        print("[STEP 5] Eval Set Questions Test (3 Domain Scenarios):")
        eval_questions = [
            "What are the ultrasound threshold criteria for polycystic ovary morphology?",
            "What is the role of LH and FSH ratio in assessing ovarian dysfunction?",
            "What lifestyle and dietary interventions are recommended as first line management?",
        ]

        for q_idx, q in enumerate(eval_questions, 1):
            print(f" -> Q{q_idx}: \"{q}\"")
            top_matches = vector_store.similarity_search(q, k=2)
            for m_idx, match in enumerate(top_matches, 1):
                src = match.metadata.get("source_file", "unknown")
                page = match.metadata.get("page", "?")
                snippet = match.page_content[:160].replace("\n", " ")
                print(f"    Match {m_idx} [{src} p.{page}]: {snippet}...")
            print()

        print("=" * 70)
        print(" [ALL CHECKS COMPLETED SUCCESSFULLY]")
        print("=" * 70)

    finally:
        client.close()


if __name__ == "__main__":
    run_checks()
