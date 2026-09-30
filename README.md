# PCOS Care Navigator 🩺

A **multi-agent AI system** for Polycystic Ovary Syndrome (PCOS) clinical decision support, combining:
- **RAG + CRAG** — Retrieval-Augmented Generation with self-correcting query rewriting over authoritative PCOS guidelines
- **Tabular ML** — XGBoost/LightGBM/CatBoost ensemble + SHAP explanations for lab-based risk scoring
- **Computer Vision** — ConvNeXt ultrasound image classifier for polycystic morphology detection
- **LangGraph** — Stateful multi-agent orchestration with semantic intent routing
- **Neo4j KG** — Structured PCOS ontology graph for symptom→criterion traversal
- **FastAPI** — REST API with full Pydantic-validated endpoints

---

## Architecture

```
START
  │
  ▼
[supervisor] ──► (Semantic Router)
  │
  ├──► "guideline"    ──► [CRAG pipeline] ──────────────────────────────┐
  ├──► "clinical"     ──► [tabular risk model + SHAP] ──────────────────┤
  ├──► "imaging"      ──► [ConvNeXt classifier] ─────────────────────────┤
  ├──► "hybrid"       ──► [clinical] ──► [imaging] ──► [CRAG] ───────────┤
  │                                                                      ▼
  └──► "request_data" ──► structured clarification ──► END      [final_synthesis] ──► END

CRAG Sub-flow:
  [retrieve] ──► [grade] ──► pass? ──► [synthesis]
                    │
                  fail? ──► [rewrite query] ──► [retrieve]  (max 1 retry)
```

---

## Prerequisites

| Tool | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.11 |
| Weaviate Cloud | Any | Free sandbox at [weaviate.io](https://weaviate.io) |
| Neo4j | 5.x | Optional — only for KG features |
| Groq API Key | — | [console.groq.com](https://console.groq.com) |

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/SIYAM1809/PCOS-RAG-Agent.git
cd PCOS-RAG-Agent
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env   # then edit .env
```

Required `.env` keys:

```dotenv
# Groq LLM API
GROQ_API_KEY=your_groq_api_key_here

# Weaviate Cloud vector store
WEAVIATE_URL=your-cluster.weaviate.network
WEAVIATE_API_KEY=your_weaviate_api_key_here

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=pcos-care-navigator

# Neo4j knowledge graph (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

### 4. Download guideline PDFs

Place the following PDFs in `data/guidelines/`:

| File | Source |
|---|---|
| `monash_2023_international_guideline.pdf` | [Monash PCOS Guideline](https://www.monash.edu/medicine/sphpm/mchri/pcos/guideline) |
| `rotterdam_2003_consensus.pdf` | [ESHRE/ASRM 2004 Consensus](https://pubmed.ncbi.nlm.nih.gov/14711538/) |
| `endocrine_society_2013_guideline.pdf` | [Endocrine Society](https://academic.oup.com/jcem/article/98/12/4565/2833703) |

### 5. Ingest guidelines into Weaviate

```bash
python -m retrieval.ingest
```

### 6. (Optional) Build ML models

Place trained model artifacts in `models/`:

| File | Type | Used by |
|---|---|---|
| `pcos_ensemble_model.joblib` | Voting classifier | `ClinicalAgent` |
| `convnext_pcos_small.pt` | PyTorch ConvNeXt | `ImagingAgent` |

> **Without models**: Both agents automatically fall back to clinically calibrated rule-based scoring (clinical) and pixel-intensity heuristics (imaging).

---

## Running

### Command-line query

```bash
# Guideline question
python -m graph.build_graph "What are the Rotterdam criteria for PCOS?"

# Full evaluation suite (11 test cases)
python -m graph.build_graph --eval
```

### API server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs at: **http://localhost:8000/docs**

### Example API requests

```bash
# Health check
curl http://localhost:8000/health

# Guideline query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the Rotterdam criteria for PCOS?"}'

# Clinical risk scoring
curl -X POST http://localhost:8000/clinical \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "lh": 18.0, "fsh": 6.0, "lh_fsh_ratio": 3.0,
      "cycle_length_days": 48, "hirsutism_score": 8, "bmi": 29.5
    }
  }'

# Ultrasound classification
curl -X POST http://localhost:8000/imaging \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/ultrasound.png"}'
```

### RAGAS evaluation

```bash
python -m eval.ragas_eval
```

Results saved to:
- `eval/ragas_scores_stage_b.csv` — per-case scores
- `eval/ragas_summary.md` — aggregate report

### Knowledge Graph

```bash
# Build the PCOS ontology graph in Neo4j
python -m retrieval.kg_ingest

# Run sample traversal queries
python -m retrieval.kg_ingest --query
```

---

## Project Structure

```
PCOS-RAG-Agent/
├── agents/
│   ├── supervisor.py          # LLM semantic intent router
│   ├── clinical_agent.py      # Tabular ensemble risk scorer + SHAP
│   └── imaging_agent.py       # ConvNeXt ultrasound classifier
│
├── graph/
│   └── build_graph.py         # LangGraph multi-agent state machine
│
├── retrieval/
│   ├── ingest.py              # PDF → Weaviate vector store pipeline
│   └── kg_ingest.py           # Neo4j PCOS knowledge graph builder
│
├── api/
│   └── main.py                # FastAPI REST endpoints
│
├── eval/
│   ├── ragas_dataset.py       # 7 benchmark test cases + ground truths
│   ├── ragas_eval.py          # RAGAS quantitative scoring script
│   ├── ragas_scores_stage_b.csv
│   └── ragas_summary.md       # Auto-generated benchmark report
│
├── data/
│   └── guidelines/            # Place PCOS guideline PDFs here
│
├── models/                    # Place trained .joblib and .pt artifacts here
├── requirements.txt
└── .env                       # API keys and connection strings
```

---

## Evaluation Results

Run `python -m eval.ragas_eval` to populate this table with live scores.

| Metric | Overall (N=7) | In-Domain (Cases 1–6) |
|---|---|---|
| Faithfulness | — | — |
| Context Precision | — | — |
| Context Recall | — | — |
| Answer Relevancy | — | — |

See [`eval/ragas_summary.md`](eval/ragas_summary.md) for the full per-case breakdown.

---

## Route Guide

| Query Type | Required Inputs | Route | Agents Used |
|---|---|---|---|
| Guideline / criteria questions | None | `guideline` | SupervisorAgent → CRAG → LLM |
| Lab value / risk explanation | `patient_data` | `clinical` | ClinicalAgent (ML + SHAP) |
| Ultrasound interpretation | `image_path` | `imaging` | ImagingAgent (ConvNeXt) |
| Full clinical picture + guidelines | `patient_data` + `image_path` | `hybrid` | All three agents |
| Missing required inputs | — | `request_data` | Structured clarification message |

---

## License

MIT License — see [LICENSE](LICENSE).
