import requests
from langchain_core.tools import tool
from app.config import SERPER_API_KEY


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information about a travel destination —
    attractions, local tips, safety, weather patterns, seasonal events, visa
    requirements, etc. Pass a specific, focused query string.
    """
    if not SERPER_API_KEY:
        return (
            "[web_search unavailable: SERPER_API_KEY not set. "
            "Sign up at serper.dev and add the key to .env]"
        )

    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"[web_search error: {e}]"

    lines = []
    if "answerBox" in data:
        ab = data["answerBox"]
        lines.append(f"Answer: {ab.get('answer') or ab.get('snippet', '')}")

    for item in data.get("organic", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"- {title}: {snippet} ({link})")

    return "\n".join(lines) if lines else "No results found."
