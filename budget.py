import json
from typing import Optional
from langchain_core.tools import tool

# Fallback only — used when the traveler does not specify their own split.
DEFAULT_WEIGHTS = {
    "lodging": 0.35,
    "food": 0.25,
    "activities": 0.20,
    "local_transport": 0.10,
    "buffer": 0.10,
}


@tool
def allocate_budget(
    total_budget: float,
    num_days: int,
    num_travelers: int,
    lodging_pct: Optional[float] = None,
    food_pct: Optional[float] = None,
    activities_pct: Optional[float] = None,
    local_transport_pct: Optional[float] = None,
    buffer_pct: Optional[float] = None,
) -> str:
    """Allocate a total trip budget across categories: lodging, food,
    activities, local_transport, buffer.

    The category percentages should come from the TRAVELER'S stated
    preference (passed in via the travel request as `budget_weights`), NOT
    invented by the agent. Pass each *_pct as a fraction (e.g. 0.35 for
    35%). If the traveler did not specify a split, omit all *_pct
    arguments and this tool will fall back to reasonable defaults —
    but the fallback must only be used when the human genuinely gave no
    preference. For a one-day trip the default split sets lodging to zero.

    Returns a JSON string with per-category amounts, the weights actually
    used, and whether those weights were human-supplied or defaulted.
    """
    if num_days <= 0:
        num_days = 1

    supplied = {
        "lodging": lodging_pct,
        "food": food_pct,
        "activities": activities_pct,
        "local_transport": local_transport_pct,
        "buffer": buffer_pct,
    }
    human_supplied = any(v is not None for v in supplied.values())

    # A one-day trip has no overnight stay, so by default nothing is set aside for
    # lodging; the other categories share the whole budget. (A split the traveler
    # typed in themselves is never changed.)
    base = dict(DEFAULT_WEIGHTS)
    if num_days == 1:
        base["lodging"] = 0.0

    if human_supplied:
        # Fill any category the human left blank with its default, then
        # normalize everything to sum to 1.0 so partial input still works.
        weights = {
            k: (v if v is not None else base[k])
            for k, v in supplied.items()
        }
        total_weight = sum(weights.values()) or 1.0
        weights = {k: v / total_weight for k, v in weights.items()}
        source = "human_supplied"
    else:
        total_weight = sum(base.values()) or 1.0
        weights = {k: v / total_weight for k, v in base.items()}
        source = "default_fallback"

    allocation = {
        category: round(total_budget * pct, 2) for category, pct in weights.items()
    }
    allocation["weights_used"] = {k: round(v, 4) for k, v in weights.items()}
    allocation["weights_source"] = source
    allocation["per_day_total"] = round(total_budget / num_days, 2)
    allocation["per_traveler_total"] = round(total_budget / max(num_travelers, 1), 2)
    allocation["num_days"] = num_days
    allocation["num_travelers"] = num_travelers

    return json.dumps(allocation, indent=2)