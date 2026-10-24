import uuid
import logging
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.auth import (router as auth_router, current_user, init_db,
                      record_plan, owns_plan, plans_today, MAX_PLANS_PER_DAY,
                      list_plans, set_status, delete_plan)
from pathlib import Path

from app.graph import travel_graph
from app.state import (
    TravelRequest, PlanCreatedResponse, PlanStatusResponse,
    ReviewRequest, FinalPlanResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_planner")

app = FastAPI(
    title="AI Travel Planner",
    description="Multi-agent travel planning system with human-in-the-loop approval.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

init_db()
app.include_router(auth_router)


def _config(plan_id: str) -> dict:
    return {"configurable": {"thread_id": plan_id}}


def _get_state_or_404(plan_id: str, user: dict):
    # A customer can only open plans they created; others look like "not found".
    if not owns_plan(plan_id, user["id"]):
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    config = _config(plan_id)
    snapshot = travel_graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    return snapshot


@app.post("/plan", response_model=PlanCreatedResponse, status_code=201)
def create_plan(request: TravelRequest, user: dict = Depends(current_user)):
    """Submit a new travel request. Runs the graph through research ->
    itinerary planning, then pauses for human review."""
    if plans_today(user["id"]) >= MAX_PLANS_PER_DAY:
        raise HTTPException(status_code=429, detail=f"Daily limit of {MAX_PLANS_PER_DAY} plans reached. Try again tomorrow.")
    plan_id = str(uuid.uuid4())
    config = _config(plan_id)

    try:
        travel_graph.invoke({"request": request.model_dump()}, config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Plan creation failed")
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}")

    snapshot = travel_graph.get_state(config)
    status = snapshot.values.get("status", "unknown")
    record_plan(plan_id, user["id"], request, status)
    return PlanCreatedResponse(plan_id=plan_id, status=status)


@app.get("/plan/{plan_id}", response_model=PlanStatusResponse)
def get_plan(plan_id: str, user: dict = Depends(current_user)):
    """Get current workflow stage and draft plan for review."""
    snapshot = _get_state_or_404(plan_id, user)
    values = snapshot.values
    return PlanStatusResponse(
        plan_id=plan_id,
        status=values.get("status", "unknown"),
        draft_itinerary=values.get("draft_itinerary"),
        research_notes=values.get("research_notes"),
        weather_outlook=values.get("weather_outlook"),
        feedback_history=values.get("feedback_history", []),
    )


@app.post("/plan/{plan_id}/review", response_model=PlanStatusResponse)
def review_plan(plan_id: str, review: ReviewRequest, user: dict = Depends(current_user)):
    """Submit HITL feedback: approve, reject (with feedback), or modify."""
    snapshot = _get_state_or_404(plan_id, user)
    config = _config(plan_id)

    if snapshot.values.get("status") not in ("pending_review",):
        raise HTTPException(
            status_code=409,
            detail=f"Plan is not awaiting review (current status: "
                   f"{snapshot.values.get('status')})",
        )

    if review.decision in ("reject", "modify") and not review.feedback:
        raise HTTPException(
            status_code=422,
            detail="`feedback` is required for reject/modify decisions",
        )

    # Inject the human decision into the persisted state, then resume the
    # graph from the interrupt point (pass None as input to continue).
    travel_graph.update_state(
        config,
        {
            "review_decision": review.decision,
            "review_feedback": review.feedback,
            "review_target": review.target,
            "status": "revising" if review.decision != "approve" else "pending_review",
        },
    )

    try:
        travel_graph.invoke(None, config)
    except Exception as e:
        logger.exception("Resume after review failed")
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}")

    snapshot = travel_graph.get_state(config)
    values = snapshot.values
    set_status(plan_id, values.get("status"))
    return PlanStatusResponse(
        plan_id=plan_id,
        status=values.get("status", "unknown"),
        draft_itinerary=values.get("draft_itinerary"),
        research_notes=values.get("research_notes"),
        weather_outlook=values.get("weather_outlook"),
        feedback_history=values.get("feedback_history", []),
    )


@app.get("/plan/{plan_id}/final", response_model=FinalPlanResponse)
def get_final_plan(plan_id: str, user: dict = Depends(current_user)):
    """Retrieve the finalized plan. Only available after approval."""
    snapshot = _get_state_or_404(plan_id, user)
    values = snapshot.values
    if values.get("status") != "finalized":
        raise HTTPException(
            status_code=409,
            detail=f"Plan not yet finalized (current status: {values.get('status')})",
        )
    return FinalPlanResponse(
        plan_id=plan_id, status=values["status"], final_plan=values["final_plan"],
    )


@app.get("/my/plans")
def my_plans(user: dict = Depends(current_user)):
    """The signed-in customer's plans, newest first, each with its current status.
    A finalized plan never changes, so its status is remembered; the others are re-read
    from the workflow so the badge on the page is always right."""
    plans = list_plans(user["id"])
    for p in plans:
        if p["status"] == "finalized":
            continue
        try:
            snap = travel_graph.get_state(_config(p["plan_id"]))
            live = snap.values.get("status") if snap and snap.values else None
        except Exception:
            live = None
        if live:
            if live != p["status"]:
                set_status(p["plan_id"], live)
            p["status"] = live
    return {"plans": plans}


@app.delete("/my/plans/{plan_id}")
def delete_my_plan(plan_id: str, user: dict = Depends(current_user)):
    """Remove a plan from the customer's list (it can no longer be opened)."""
    if not delete_plan(plan_id, user["id"]):
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    try:                                   # free the stored workflow data too, when this library version supports it
        saver = getattr(travel_graph, "checkpointer", None)
        if saver is not None and hasattr(saver, "delete_thread"):
            saver.delete_thread(plan_id)
    except Exception:
        logger.warning("Could not purge stored data for plan %s", plan_id)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Frontend: serve the single-page UI at http://localhost:8000/ ---
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")