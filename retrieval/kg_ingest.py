"""
Knowledge Graph Ingestion — retrieval/kg_ingest.py.

Builds a Neo4j property graph encoding the PCOS clinical ontology:
  - Entities : Symptom, DiagnosticCriterion, Guideline, Treatment, Biomarker
  - Relations: HAS_SYMPTOM, SUPPORTS_CRITERION, REFERENCED_IN, RECOMMENDS, INDICATES

The graph enables structured traversal queries (e.g. "which criteria does
hirsutism support?") that complement the vector-similarity RAG pipeline.

Usage:
    python -m retrieval.kg_ingest           # Build full KG
    python -m retrieval.kg_ingest --query   # Run sample queries
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()


# =====================================================================
# PCOS Clinical Ontology — Domain Knowledge
# =====================================================================

PCOS_ONTOLOGY: Dict[str, Any] = {
    "guidelines": [
        {"id": "rotterdam_2003", "name": "Rotterdam 2003 PCOS Consensus", "year": 2003,
         "body": "ESHRE/ASRM"},
        {"id": "monash_2023",    "name": "Monash 2023 International PCOS Guideline", "year": 2023,
         "body": "Monash University"},
        {"id": "endocrine_2013", "name": "Endocrine Society 2013 Clinical Practice Guideline", "year": 2013,
         "body": "Endocrine Society"},
    ],

    "diagnostic_criteria": [
        {"id": "oligo_anovulation",    "name": "Oligo/Anovulation",
         "description": "Irregular or absent ovulation (cycle > 35 days or < 8 cycles/year)"},
        {"id": "hyperandrogenism",     "name": "Hyperandrogenism",
         "description": "Clinical (hirsutism, acne, androgenic alopecia) or biochemical (elevated androgens)"},
        {"id": "polycystic_morphology","name": "Polycystic Ovarian Morphology (PCOM)",
         "description": "FNPO >= 20 or ovarian volume >= 10 mL on ultrasound (Monash 2023)"},
    ],

    "symptoms": [
        {"id": "irregular_periods",   "name": "Irregular or skipped menstrual periods",
         "clinical_term": "oligomenorrhea / amenorrhea",   "criteria": ["oligo_anovulation"]},
        {"id": "hirsutism",           "name": "Excess facial/body hair growth",
         "clinical_term": "hirsutism",                     "criteria": ["hyperandrogenism"]},
        {"id": "acne",                "name": "Persistent acne",
         "clinical_term": "acne vulgaris (androgen-driven)","criteria": ["hyperandrogenism"]},
        {"id": "alopecia",            "name": "Scalp hair thinning",
         "clinical_term": "androgenic alopecia",           "criteria": ["hyperandrogenism"]},
        {"id": "ovarian_cysts",       "name": "Small cysts on ovaries (ultrasound)",
         "clinical_term": "polycystic ovarian morphology", "criteria": ["polycystic_morphology"]},
        {"id": "weight_gain",         "name": "Unexplained weight gain / difficulty losing weight",
         "clinical_term": "metabolic dysfunction / insulin resistance", "criteria": []},
        {"id": "infertility",         "name": "Difficulty conceiving",
         "clinical_term": "anovulatory infertility",       "criteria": ["oligo_anovulation"]},
    ],

    "biomarkers": [
        {"id": "lh_fsh_ratio",      "name": "LH:FSH ratio",
         "threshold": ">= 2:1",    "significance": "Elevated LH relative to FSH; neuroendocrine marker",
         "criteria": ["oligo_anovulation", "hyperandrogenism"]},
        {"id": "testosterone",      "name": "Free / Total Testosterone",
         "threshold": "Elevated",  "significance": "Biochemical hyperandrogenism marker",
         "criteria": ["hyperandrogenism"]},
        {"id": "amh",               "name": "Anti-Müllerian Hormone (AMH)",
         "threshold": "> 3.5 ng/mL","significance": "Reflects antral follicle count; elevated in PCOS",
         "criteria": ["polycystic_morphology"]},
        {"id": "fasting_glucose",   "name": "Fasting Plasma Glucose / HOMA-IR",
         "threshold": ">= 100 mg/dL","significance": "Insulin resistance screening",
         "criteria": []},
    ],

    "treatments": [
        {"id": "lifestyle",         "name": "Lifestyle Intervention",
         "line": "first",          "description": "Diet, exercise, behavioural modification",
         "guidelines": ["monash_2023", "rotterdam_2003"]},
        {"id": "ocp",              "name": "Combined Oral Contraceptive Pill (OCP)",
         "line": "second",         "description": "Regulates cycle, reduces androgen excess",
         "guidelines": ["monash_2023", "endocrine_2013"]},
        {"id": "metformin",        "name": "Metformin",
         "line": "second",         "description": "Insulin sensitiser; improves metabolic profile",
         "guidelines": ["endocrine_2013"]},
        {"id": "letrozole",        "name": "Letrozole (Ovulation Induction)",
         "line": "second",         "description": "First-line pharmacological ovulation induction",
         "guidelines": ["monash_2023"]},
    ],
}


# =====================================================================
# Neo4j Client Helper
# =====================================================================

def _get_driver():
    """Returns a Neo4j driver using credentials from .env."""
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j Python driver is required. Install with: pip install neo4j"
        ) from e

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    if not password:
        raise ValueError(
            "NEO4J_PASSWORD is not set in .env. "
            "Add NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD to your .env file."
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


# =====================================================================
# Graph Build Functions
# =====================================================================

def _clear_graph(session) -> None:
    """Removes all nodes and relationships (fresh ingest)."""
    session.run("MATCH (n) DETACH DELETE n")
    print("[KG] Graph cleared.")


def _create_constraints(session) -> None:
    """Creates uniqueness constraints for all entity types."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Guideline)          REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:DiagnosticCriterion) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom)            REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Biomarker)          REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment)          REQUIRE t.id IS UNIQUE",
    ]
    for stmt in constraints:
        session.run(stmt)
    print("[KG] Constraints created.")


def _ingest_guidelines(session, data: List[Dict]) -> None:
    for g in data:
        session.run(
            "MERGE (n:Guideline {id: $id}) "
            "SET n.name = $name, n.year = $year, n.body = $body",
            **g
        )
    print(f"[KG] Ingested {len(data)} Guideline nodes.")


def _ingest_criteria(session, data: List[Dict]) -> None:
    for c in data:
        session.run(
            "MERGE (n:DiagnosticCriterion {id: $id}) "
            "SET n.name = $name, n.description = $description",
            **c
        )
    print(f"[KG] Ingested {len(data)} DiagnosticCriterion nodes.")


def _ingest_symptoms(session, data: List[Dict]) -> None:
    for s in data:
        session.run(
            "MERGE (n:Symptom {id: $id}) "
            "SET n.name = $name, n.clinical_term = $clinical_term",
            id=s["id"], name=s["name"], clinical_term=s["clinical_term"]
        )
        for crit_id in s.get("criteria", []):
            session.run(
                "MATCH (s:Symptom {id: $sid}), (c:DiagnosticCriterion {id: $cid}) "
                "MERGE (s)-[:SUPPORTS_CRITERION]->(c)",
                sid=s["id"], cid=crit_id
            )
    print(f"[KG] Ingested {len(data)} Symptom nodes with SUPPORTS_CRITERION edges.")


def _ingest_biomarkers(session, data: List[Dict]) -> None:
    for b in data:
        session.run(
            "MERGE (n:Biomarker {id: $id}) "
            "SET n.name = $name, n.threshold = $threshold, n.significance = $significance",
            id=b["id"], name=b["name"],
            threshold=b["threshold"], significance=b["significance"]
        )
        for crit_id in b.get("criteria", []):
            session.run(
                "MATCH (b:Biomarker {id: $bid}), (c:DiagnosticCriterion {id: $cid}) "
                "MERGE (b)-[:INDICATES]->(c)",
                bid=b["id"], cid=crit_id
            )
    print(f"[KG] Ingested {len(data)} Biomarker nodes with INDICATES edges.")


def _ingest_treatments(session, data: List[Dict]) -> None:
    for t in data:
        session.run(
            "MERGE (n:Treatment {id: $id}) "
            "SET n.name = $name, n.line = $line, n.description = $description",
            id=t["id"], name=t["name"],
            line=t["line"], description=t["description"]
        )
        for g_id in t.get("guidelines", []):
            session.run(
                "MATCH (t:Treatment {id: $tid}), (g:Guideline {id: $gid}) "
                "MERGE (g)-[:RECOMMENDS]->(t)",
                tid=t["id"], gid=g_id
            )
    print(f"[KG] Ingested {len(data)} Treatment nodes with RECOMMENDS edges.")


# =====================================================================
# Public Interface
# =====================================================================

def build_kg() -> None:
    """
    Builds the full PCOS knowledge graph in Neo4j.
    Drops and recreates the graph on each run for idempotent ingestion.
    """
    print("=" * 60)
    print("  PCOS Knowledge Graph Ingestion")
    print("=" * 60)

    driver = _get_driver()
    print(f"[KG] Connected to Neo4j at {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")

    with driver.session() as session:
        _clear_graph(session)
        _create_constraints(session)
        _ingest_guidelines(session,  PCOS_ONTOLOGY["guidelines"])
        _ingest_criteria(session,    PCOS_ONTOLOGY["diagnostic_criteria"])
        _ingest_symptoms(session,    PCOS_ONTOLOGY["symptoms"])
        _ingest_biomarkers(session,  PCOS_ONTOLOGY["biomarkers"])
        _ingest_treatments(session,  PCOS_ONTOLOGY["treatments"])

    driver.close()

    total_nodes = (
        len(PCOS_ONTOLOGY["guidelines"])
        + len(PCOS_ONTOLOGY["diagnostic_criteria"])
        + len(PCOS_ONTOLOGY["symptoms"])
        + len(PCOS_ONTOLOGY["biomarkers"])
        + len(PCOS_ONTOLOGY["treatments"])
    )
    print(f"\n[KG] Build complete. {total_nodes} nodes ingested.")
    print("=" * 60)


def query_kg(symptom_name: str) -> List[Dict[str, Any]]:
    """
    Retrieves the diagnostic criteria supported by a given symptom or biomarker.

    Args:
        symptom_name: Plain-English symptom name or clinical term (case-insensitive partial match).

    Returns:
        List of dicts with keys: symptom, clinical_term, criterion, criterion_description.
    """
    driver = _get_driver()
    results = []

    with driver.session() as session:
        # Symptom → Criterion lookup
        records = session.run(
            """
            MATCH (s:Symptom)-[:SUPPORTS_CRITERION]->(c:DiagnosticCriterion)
            WHERE toLower(s.name) CONTAINS toLower($name)
               OR toLower(s.clinical_term) CONTAINS toLower($name)
            RETURN s.name AS symptom, s.clinical_term AS clinical_term,
                   c.name AS criterion, c.description AS criterion_description
            """,
            name=symptom_name
        )
        for r in records:
            results.append(dict(r))

        # Biomarker → Criterion lookup
        records = session.run(
            """
            MATCH (b:Biomarker)-[:INDICATES]->(c:DiagnosticCriterion)
            WHERE toLower(b.name) CONTAINS toLower($name)
            RETURN b.name AS symptom, b.significance AS clinical_term,
                   c.name AS criterion, c.description AS criterion_description
            """,
            name=symptom_name
        )
        for r in records:
            results.append(dict(r))

    driver.close()
    return results


# =====================================================================
# CLI
# =====================================================================

def _run_sample_queries(driver) -> None:
    """Demonstrates graph traversal capabilities with 3 sample queries."""
    print("\n[KG] Running sample queries...")

    with driver.session() as session:
        # Q1: Which criteria does hirsutism support?
        print("\n[Q1] Criteria supported by 'hirsutism':")
        res = session.run(
            "MATCH (s:Symptom {id: 'hirsutism'})-[:SUPPORTS_CRITERION]->(c) RETURN c.name"
        )
        for r in res:
            print(f"  • {r['c.name']}")

        # Q2: First-line treatments referenced in Monash 2023
        print("\n[Q2] Treatments recommended in Monash 2023:")
        res = session.run(
            "MATCH (g:Guideline {id: 'monash_2023'})-[:RECOMMENDS]->(t) "
            "RETURN t.name, t.line"
        )
        for r in res:
            print(f"  • {r['t.name']} (Line: {r['t.line']})")

        # Q3: All biomarkers indicating Hyperandrogenism
        print("\n[Q3] Biomarkers indicating Hyperandrogenism:")
        res = session.run(
            "MATCH (b:Biomarker)-[:INDICATES]->(c:DiagnosticCriterion {id: 'hyperandrogenism'}) "
            "RETURN b.name, b.threshold"
        )
        for r in res:
            print(f"  • {r['b.name']} (threshold: {r['b.threshold']})")


if __name__ == "__main__":
    if "--query" in sys.argv:
        driver = _get_driver()
        _run_sample_queries(driver)
        driver.close()
    else:
        build_kg()
        driver = _get_driver()
        _run_sample_queries(driver)
        driver.close()
