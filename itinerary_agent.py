import json
import re
from datetime import date, timedelta
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from app.config import get_llm
from app.tools.budget import allocate_budget
from app.tools.packing import generate_packing_list

ITINERARY_SYSTEM_PROMPT = """You are an itinerary planning specialist. Given a
destination research brief and the traveler's constraints, build a
day-by-day trip plan. Use your tools to allocate the budget and generate a
packing list.

When calling allocate_budget: if the traveler specified budget_weights,
you MUST pass those exact values as the *_pct arguments. Do NOT invent or
adjust the split yourself — the human's stated preference takes priority.
Only omit the *_pct arguments if the traveler gave no preference at all.

When you write budget_allocation, copy the allocate_budget tool result
through FAITHFULLY. You MUST include the "weights_used" and
"weights_source" fields exactly as the tool returned them — never drop or
alter them. "weights_source" is how a reviewer verifies whether the human's
split or the default fallback was used, so it must survive into your output.

For "weather_summary", pull the actual forecast details (temperatures,
rain chances) from the research brief's Weather section — do not write a
vague statement like "check the weather"; include the real numbers you
were given.

You MUST respond with ONLY a single valid JSON object (no markdown fences,
no prose before or after) with this exact shape:

{
  "destination": string,
  "dates": {"start": string, "end": string},
  "travelers": number,
  "weather_summary": string,
  "budget_allocation": {
    "lodging": number,
    "food": number,
    "activities": number,
    "local_transport": number,
    "buffer": number,
    "weights_used": object,
    "weights_source": string
  },
  "days": [
    {"day": number, "date": string, "theme": string,
     "activities": [{"time": string, "activity": string, "notes": string}]}
  ],
  "packing_list": [string, ...],
  "tips": [string, ...]
}
"""


def build_itinerary_agent():
    return create_react_agent(get_llm(), tools=[allocate_budget, generate_packing_list])


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _last_tool_output(messages, tool_name: str) -> dict | None:
    """Return the parsed JSON payload from the most recent successful call to
    `tool_name` in the agent's message history, or None if it was never
    called / never returned parseable JSON. Reads the ToolMessage directly
    from the transcript, so it does NOT depend on the LLM choosing to echo
    the tool's fields back to us.
    """
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) == tool_name:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def _expected_dates(start_date: str, end_date: str) -> list[str]:
    """Every calendar date from start to end INCLUSIVE, as ISO strings."""
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (ValueError, TypeError):
        return [start_date]
    span = (end - start).days
    if span < 0:
        return [start_date]
    return [(start + timedelta(days=i)).isoformat() for i in range(span + 1)]


def _day_count(d: dict) -> int:
    days = d.get("days")
    return len(days) if isinstance(days, list) else -1


def run_itinerary(state: dict) -> dict:
    """LangGraph node: runs the itinerary planner agent and writes draft_itinerary."""
    req = state["request"]
    revision_note = ""
    if state.get("review_decision") in ("reject", "modify") and state.get("review_target") in (None, "itinerary"):
        revision_note = (
            f"\n\nIMPORTANT — revise the PREVIOUS DRAFT below based on this feedback:\n"
            f"Feedback: {state.get('review_feedback')}\n"
            f"Previous draft: {json.dumps(state.get('draft_itinerary', {}))}"
        )

    weights = req.get("budget_weights") or {}
    if any(v is not None for v in weights.values()):
        weights_line = (
            "- Budget split (human-specified, use these exact *_pct values "
            f"when calling allocate_budget): {json.dumps(weights)}"
        )
    else:
        weights_line = (
            "- Budget split: not specified by traveler — omit *_pct "
            "arguments and let allocate_budget use its defaults"
        )

    trip_dates = _expected_dates(req["start_date"], req["end_date"])
    num_days = len(trip_dates)
    days_line = (
        f"- Trip length: {num_days} day(s). The \"days\" array MUST contain "
        f"EXACTLY {num_days} objects — one per date, in order, covering every "
        f"one of these dates: {', '.join(trip_dates)}. Do NOT stop early, "
        f"summarize, or collapse multiple dates into one entry; every listed "
        f"date needs its own day object with real activities."
    )

    # The planner works on the middle of the budget range (this is also what validation.py checks).
    plan_total = round((req["budget_min"] + req["budget_max"]) / 2)
    budget_line = (
        f"\n- Budget for the trip: ₹{plan_total} (the middle of the range). "
        f"Call allocate_budget with total_budget exactly {plan_total}."
    )

    prompt = (
        f"{ITINERARY_SYSTEM_PROMPT}\n\n"
        f"Research brief:\n{state.get('research_notes', '')}\n\n"
        f"Travel request:\n"
        f"- Destination: {req['destination']}\n"
        f"- Dates: {req['start_date']} to {req['end_date']}\n"
        f"- Budget: {req['budget_min']}-{req['budget_max']} (INR)\n"
        f"- Interests: {', '.join(req.get('interests', [])) or 'general sightseeing'}\n"
        f"- Travelers: {req['num_travelers']}\n"
        f"{days_line}\n"
        f"{weights_line}"
        f"{budget_line}"
        f"{revision_note}"
    )

    agent = build_itinerary_agent()
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    messages = result["messages"]
    raw = messages[-1].content

    try:
        draft = _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        draft = {"error": "Could not parse structured itinerary", "raw_output": raw}

    # Bounded single retry if the model returned the wrong number of days.
    if "error" not in draft and _day_count(draft) != num_days:
        correction = HumanMessage(content=(
            f"Your itinerary had {_day_count(draft)} day object(s), but this is a "
            f"{num_days}-day trip. Regenerate the COMPLETE JSON object with exactly "
            f"{num_days} entries in \"days\", one for each of these dates in order: "
            f"{', '.join(trip_dates)}. Keep everything else consistent. "
            f"Respond with ONLY the JSON object."
        ))
        result = agent.invoke({"messages": messages + [correction]})
        messages = result["messages"]
        raw = messages[-1].content
        try:
            draft = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            draft = {"error": "Could not parse structured itinerary", "raw_output": raw}

    if "error" not in draft and isinstance(draft, dict):
        # Budget reliability merge: the tool output is the source of truth
        # for weights_used / weights_source, regardless of what the LLM wrote.
        tool_out = _last_tool_output(messages, "allocate_budget")
        if tool_out is not None:
            merged = dict(draft.get("budget_allocation") or {})
            for key in ("weights_used", "weights_source"):
                if key in tool_out:
                    merged[key] = tool_out[key]
            draft["budget_allocation"] = merged

        # Weather reliability backstop: if the LLM left weather_summary
        # empty/missing, fall back to the raw weather tool output the
        # research agent captured — so weather always survives into the
        # final plan when the tool call succeeded upstream.
        if not draft.get("weather_summary"):
            fallback_weather = state.get("weather_outlook")
            if fallback_weather:
                draft["weather_summary"] = fallback_weather

    history = state.get("feedback_history", [])
    if state.get("review_decision") in ("reject", "modify"):
        history = history + [{
            "decision": state["review_decision"],
            "feedback": state.get("review_feedback", ""),
            "target": state.get("review_target", "itinerary"),
        }]

    return {
        "draft_itinerary": draft,
        "status": "pending_review",
        "review_decision": None,
        "review_feedback": None,
        "feedback_history": history,
        "revision_count": state.get("revision_count", 0) + (1 if history else 0),
    }