import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.state import WorkflowState
from app.agents.research_agent import run_research
from app.agents.itinerary_agent import run_itinerary
from app.config import DATABASE_PATH
from app.validation import validate_travel_request


def validate_request(state: WorkflowState) -> dict:
    req = state["request"]
    required = ["destination", "start_date", "end_date", "budget_min",
                "budget_max", "num_travelers"]
    missing = [f for f in required if f not in req or req[f] in (None, "")]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    errors = validate_travel_request(req)
    if errors:
        raise ValueError("Invalid request: " + "; ".join(errors))

    return {"status": "researching", "feedback_history": [], "revision_count": 0}


def human_review(state: WorkflowState) -> dict:
    """LangGraph node: human-in-the-loop"""
    return {}


def finalize_output(state: WorkflowState) -> dict:
    return {
        "status": "finalized",
        "final_plan": state["draft_itinerary"],
    }


def route_after_review(state: WorkflowState) -> str:
    decision = state.get("review_decision")
    if decision == "approve":
        return "finalize"
    if decision == "modify":
        return "itinerary_agent"
    if decision == "reject":
        target = state.get("review_target", "itinerary")
        return "research_agent" if target == "research" else "itinerary_agent"
    return "finalize"


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("validate", validate_request)
    graph.add_node("research_agent", run_research)
    graph.add_node("itinerary_agent", run_itinerary)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize_output)

    graph.set_entry_point("validate")
    graph.add_edge("validate", "research_agent")
    graph.add_edge("research_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "human_review")

    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "finalize": "finalize",
            "research_agent": "research_agent",
            "itinerary_agent": "itinerary_agent",
        },
    )
    graph.add_edge("finalize", END)

    os.makedirs(os.path.dirname(DATABASE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
    return compiled

travel_graph = build_graph()