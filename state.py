from typing import Optional, Literal, List, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LangGraph state (the dict threaded through every node)
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    # original request
    request: Dict[str, Any]

    # agent outputs
    research_notes: str
    weather_outlook: str
    draft_itinerary: Dict[str, Any]

    # HITL
    status: Literal[
        "researching", "planning", "pending_review", "revising",
        "approved", "finalized",
    ]
    review_decision: Optional[Literal["approve", "reject", "modify"]]
    review_feedback: Optional[str]
    review_target: Optional[Literal["research", "itinerary"]]
    feedback_history: List[Dict[str, str]]
    revision_count: int

    # final output
    final_plan: Dict[str, Any]


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------

class BudgetWeights(BaseModel):
    """Optional human-specified split of the trip budget across categories.
    Each value is a fraction (need not sum to exactly 1.0 — they are
    normalized). Any category left out falls back to a default weight,
    only for that category."""
    lodging_pct: Optional[float] = None
    food_pct: Optional[float] = None
    activities_pct: Optional[float] = None
    local_transport_pct: Optional[float] = None
    buffer_pct: Optional[float] = None


class TravelRequest(BaseModel):
    destination: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    budget_min: float
    budget_max: float
    interests: List[str] = Field(default_factory=list)
    num_travelers: int = 1
    budget_weights: Optional[BudgetWeights] = Field(
        None,
        description=(
            "Optional: how the traveler wants their budget split across "
            "lodging/food/activities/local_transport/buffer. If omitted, "
            "the system falls back to reasonable defaults."
        ),
    )


class PlanCreatedResponse(BaseModel):
    plan_id: str
    status: str


class PlanStatusResponse(BaseModel):
    plan_id: str
    status: str
    draft_itinerary: Optional[Dict[str, Any]] = None
    research_notes: Optional[str] = None
    weather_outlook: Optional[str] = None
    feedback_history: List[Dict[str, str]] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject", "modify"]
    feedback: Optional[str] = Field(
        None, description="Required for reject/modify — what to change and why"
    )
    target: Optional[Literal["research", "itinerary"]] = Field(
        "itinerary", description="Which agent should handle the revision"
    )


class FinalPlanResponse(BaseModel):
    plan_id: str
    status: str
    final_plan: Dict[str, Any]