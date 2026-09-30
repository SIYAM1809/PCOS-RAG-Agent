"""
FastAPI application — PCOS Care Navigator REST API.

Endpoints:
  GET  /health          — Liveness probe
  POST /query           — Full multi-agent pipeline (guideline / clinical / imaging / hybrid)
  POST /clinical        — Direct clinical risk score from patient lab data
  POST /imaging         — Direct ultrasound classification from image path
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph.build_graph import build_graph, _make_initial_state
from agents.clinical_agent import ClinicalAgent
from agents.imaging_agent import ImagingAgent


# =====================================================================
# Application Lifespan — resources initialized once at startup
# =====================================================================
_graph_app = None
_graph_res = None
_clinical_agent = None
_imaging_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all heavy resources at startup; clean up on shutdown."""
    global _graph_app, _graph_res, _clinical_agent, _imaging_agent
    print("[API] Initializing PCOS Care Navigator resources...")
    _graph_app, _graph_res = build_graph()
    _clinical_agent = ClinicalAgent()
    _imaging_agent = ImagingAgent()
    print("[API] All resources ready. Server is live.")
    yield
    # --- Shutdown ---
    if _graph_res:
        _graph_res.close()
    print("[API] Resources released. Server shut down.")


# =====================================================================
# FastAPI App
# =====================================================================
app = FastAPI(
    title="PCOS Care Navigator API",
    description=(
        "Multi-agent clinical decision support system for Polycystic Ovary Syndrome (PCOS). "
        "Combines RAG guideline retrieval, tabular clinical risk scoring, and ultrasound image analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request / Response Models
# =====================================================================

class PatientData(BaseModel):
    lh: Optional[float] = Field(None, description="Luteinizing Hormone (mIU/mL)")
    fsh: Optional[float] = Field(None, description="Follicle-Stimulating Hormone (mIU/mL)")
    lh_fsh_ratio: Optional[float] = Field(None, description="LH:FSH ratio (computed if not provided)")
    cycle_length_days: Optional[float] = Field(None, description="Menstrual cycle length in days")
    hirsutism_score: Optional[float] = Field(None, description="Modified Ferriman-Gallwey hirsutism score")
    bmi: Optional[float] = Field(None, description="Body Mass Index (kg/m²)")
    free_testosterone: Optional[float] = Field(None, description="Free testosterone (pg/mL)")
    amh_level: Optional[float] = Field(None, description="Anti-Müllerian hormone (ng/mL)")
    fasting_glucose: Optional[float] = Field(None, description="Fasting plasma glucose (mg/dL)")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query or clinical question")
    patient_data: Optional[PatientData] = Field(None, description="Patient lab / symptom values (for clinical/hybrid routes)")
    image_path: Optional[str] = Field(None, description="Absolute path to ultrasound image (for imaging/hybrid routes)")


class ClinicalRequest(BaseModel):
    patient_data: PatientData = Field(..., description="Patient lab / symptom values")


class ImagingRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to the ultrasound image file")


class RouteInfo(BaseModel):
    route: Optional[str]
    route_history: List[str]


class QueryResponse(BaseModel):
    final_answer: str
    route_info: RouteInfo
    clinical_result: Optional[Dict[str, Any]] = None
    imaging_result: Optional[Dict[str, Any]] = None
    guideline_sources: Optional[List[Dict[str, Any]]] = None
    missing_requirements: Optional[List[str]] = None


class ClinicalResponse(BaseModel):
    risk_score: float
    risk_label: str
    top_contributing_factors: List
    patient_summary: Dict[str, Any]


class ImagingResponse(BaseModel):
    image_path: str
    pcos_probability: float
    classification: str
    confidence: float
    morphology_features: List[str]


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    models_loaded: Dict[str, bool]


# =====================================================================
# Endpoints
# =====================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["System"],
)
def health_check():
    """Returns system health status and model availability."""
    return HealthResponse(
        status="ok",
        app="PCOS Care Navigator",
        version="1.0.0",
        models_loaded={
            "graph": _graph_app is not None,
            "clinical_agent": _clinical_agent is not None and _clinical_agent.model is not None,
            "imaging_agent": _imaging_agent is not None and _imaging_agent.model is not None,
        }
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Full multi-agent pipeline query",
    tags=["Query"],
)
def query_endpoint(request: QueryRequest):
    """
    Routes the query through the full multi-agent state machine.

    Automatically selects the appropriate pipeline based on query intent
    and available inputs (patient_data, image_path):
    - **guideline**: RAG retrieval + CRAG self-correction
    - **clinical**: Tabular ensemble risk scoring + SHAP attribution
    - **imaging**: ConvNeXt ultrasound classification
    - **hybrid**: Clinical + Imaging + Guideline synthesis
    - **request_data**: Returns structured clarification message if required inputs are missing
    """
    if _graph_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph pipeline not yet initialized. Retry in a moment."
        )

    patient_dict = request.patient_data.model_dump(exclude_none=True) if request.patient_data else None

    try:
        initial_state = _make_initial_state(
            query=request.query,
            patient_data=patient_dict,
            image_path=request.image_path,
        )
        result = _graph_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph execution error: {str(e)}"
        )

    return QueryResponse(
        final_answer=result.get("final_answer") or "",
        route_info=RouteInfo(
            route=result.get("route"),
            route_history=result.get("route_history", []),
        ),
        clinical_result=result.get("clinical_result"),
        imaging_result=result.get("imaging_result"),
        guideline_sources=result.get("guideline_sources"),
        missing_requirements=result.get("missing_requirements"),
    )


@app.post(
    "/clinical",
    response_model=ClinicalResponse,
    summary="Clinical risk scoring",
    tags=["Agents"],
)
def clinical_endpoint(request: ClinicalRequest):
    """
    Runs the tabular ensemble risk model (XGBoost/LightGBM/CatBoost + SHAP) directly
    on the provided patient lab values and returns a risk score with top contributing factors.
    """
    if _clinical_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clinical agent not yet initialized."
        )

    patient_dict = request.patient_data.model_dump(exclude_none=True)
    if not patient_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one patient data field must be provided."
        )

    try:
        result = _clinical_agent.run(patient_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical agent error: {str(e)}"
        )

    return ClinicalResponse(**result)


@app.post(
    "/imaging",
    response_model=ImagingResponse,
    summary="Ultrasound image classification",
    tags=["Agents"],
)
def imaging_endpoint(request: ImagingRequest):
    """
    Runs the ConvNeXt classifier (or pixel-intensity heuristic fallback) on the
    provided ultrasound image path and returns PCOS probability + morphology features.
    """
    if _imaging_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Imaging agent not yet initialized."
        )

    try:
        result = _imaging_agent.run(request.image_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Imaging agent error: {str(e)}"
        )

    return ImagingResponse(**result)


# =====================================================================
# Dev entrypoint
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
