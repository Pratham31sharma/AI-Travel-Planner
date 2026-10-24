from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from app.config import get_llm
from app.tools.web_search import web_search
from app.tools.weather import get_weather_outlook

RESEARCH_SYSTEM_PROMPT = """You are a destination research specialist for a travel
planning system. Given a travel request, use your tools to gather:
- Top attractions and experiences matching the traveler's interests
- Local tips (customs, safety, transport)
- Weather outlook for the travel dates (use get_weather_outlook)
- Any seasonal considerations (festivals, closures, peak/off-peak)

Be concise but concrete. Output a structured research brief in plain text
with clear sections, including a dedicated "Weather" section with the
actual forecast details (temperatures, rain chances) the tool returned —
do not paraphrase away the specific numbers. You are not building the
itinerary — just gathering the intelligence the planner will need.
"""


def build_research_agent():
    return create_react_agent(get_llm(), tools=[web_search, get_weather_outlook])


def _last_tool_text(messages, tool_name: str) -> str | None:
    """Return the raw text content of the most recent call to `tool_name`
    in the agent's message history, or None if it was never called.

    Used as a reliability backstop for weather: the LLM's prose summary
    might drop specific forecast numbers, but this always has the exact
    string the weather tool returned, independent of what the model chose
    to write in its final answer.
    """
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) == tool_name:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return content
    return None


def run_research(state: dict) -> dict:
    """LangGraph node: runs the research agent and writes research_notes
    plus a separately-captured weather_outlook."""
    req = state["request"]
    revision_note = ""
    if state.get("review_decision") == "reject" and state.get("review_target") == "research":
        revision_note = (
            f"\n\nIMPORTANT — the previous research was rejected with this feedback, "
            f"address it directly: {state.get('review_feedback')}"
        )

    prompt = (
        f"{RESEARCH_SYSTEM_PROMPT}\n\n"
        f"Travel request:\n"
        f"- Destination: {req['destination']}\n"
        f"- Dates: {req['start_date']} to {req['end_date']}\n"
        f"- Budget: {req['budget_min']}-{req['budget_max']} (INR)\n"
        f"- Interests: {', '.join(req.get('interests', [])) or 'general sightseeing'}\n"
        f"- Travelers: {req['num_travelers']}"
        f"{revision_note}"
    )

    agent = build_research_agent()
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    messages = result["messages"]
    notes = messages[-1].content

    # Reliability backstop: capture the weather tool's raw output directly,
    # rather than trusting the LLM's prose summary to preserve it faithfully.
    weather_outlook = _last_tool_text(messages, "get_weather_outlook") or ""

    return {
        "research_notes": notes,
        "weather_outlook": weather_outlook,
        "status": "planning",
    }