from datetime import date
from functools import lru_cache
import requests

MIN_REASONABLE_BUDGET_PER_TRAVELER_PER_DAY_INR = 400


def _validate_dates(req: dict, errors: list[str]) -> None:
    try:
        start = date.fromisoformat(req["start_date"])
    except (ValueError, TypeError):
        errors.append(f"start_date '{req.get('start_date')}' is not a valid YYYY-MM-DD date")
        start = None

    try:
        end = date.fromisoformat(req["end_date"])
    except (ValueError, TypeError):
        errors.append(f"end_date '{req.get('end_date')}' is not a valid YYYY-MM-DD date")
        end = None

    if start and end and end < start:
        errors.append(
            f"end_date ({req['end_date']}) is before start_date ({req['start_date']})"
        )


def _validate_budget(req: dict, errors: list[str]) -> None:
    budget_min = req.get("budget_min")
    budget_max = req.get("budget_max")

    if not isinstance(budget_min, (int, float)) or budget_min <= 0:
        errors.append(f"budget_min must be a positive number, got {budget_min!r}")
        return
    if not isinstance(budget_max, (int, float)) or budget_max <= 0:
        errors.append(f"budget_max must be a positive number, got {budget_max!r}")
        return
    if budget_max < budget_min:
        errors.append(
            f"budget_max ({budget_max}) is less than budget_min ({budget_min})"
        )
        return

    # Sanity-check against trip length / travelers, if those are valid too.
    try:
        start = date.fromisoformat(req["start_date"])
        end = date.fromisoformat(req["end_date"])
        num_days = max((end - start).days + 1, 1)
    except (ValueError, TypeError, KeyError):
        num_days = None

    travelers = req.get("num_travelers")
    numbers_ok = (
        isinstance(budget_min, (int, float)) and isinstance(budget_max, (int, float))
        and budget_min > 0 and budget_max >= budget_min
    )
    if numbers_ok and num_days and isinstance(travelers, int) and travelers > 0:
        # The planner works on the MIDDLE of the range, so that is what is checked
        # (not the lowest number, which would reject e.g. ₹2000-₹5000 for no reason).
        middle = (budget_min + budget_max) / 2
        per_day_per_traveler = middle / num_days / travelers
        if per_day_per_traveler < MIN_REASONABLE_BUDGET_PER_TRAVELER_PER_DAY_INR:
            errors.append(
                f"your budget (₹{budget_min:g} to ₹{budget_max:g}) works out to only "
                f"₹{per_day_per_traveler:.2f} per traveler per day at the middle of the range "
                f"(₹{middle:g}), which is too low to plan a realistic trip. Raise the budget, "
                f"shorten the trip or reduce the number of travelers"
            )


def _validate_travelers(req: dict, errors: list[str]) -> None:
    travelers = req.get("num_travelers")
    if not isinstance(travelers, int) or travelers < 1:
        errors.append(f"num_travelers must be a whole number >= 1, got {travelers!r}")


class GeocodingRateLimited(Exception):
    pass

@lru_cache(maxsize=512)
def _geocode_results_cached(query: str, count: int) -> tuple:
 
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "addressdetails": 1, "limit": count},
        headers={"User-Agent": "ai-travel-planner-validation/1.0"},
        timeout=4,  # was 10s — fail fast rather than blocking /plan for a long time
    )
    if resp.status_code == 429:
        raise GeocodingRateLimited("Nominatim rate limit hit (429)")
    resp.raise_for_status()

    results = []
    for r in resp.json() or []:
        address = r.get("address", {})
        results.append((
            r.get("name") or r.get("display_name"),
            address.get("country", ""),
            (address.get("country_code") or "").upper(),
        ))
    return tuple(results)


def _geocode_results(query: str, count: int = 10) -> list[dict]:

    raw = _geocode_results_cached(query, count)
    return [
        {"name": name, "country": country, "country_code": country_code}
        for name, country, country_code in raw
    ]


def _validate_destination(req: dict, errors: list[str]) -> None:

    destination = req.get("destination", "")
    if not destination:
        return

    parts = [p.strip() for p in destination.split(",")]
    city_part = parts[0]
    stated_country = parts[-1] if len(parts) >= 2 else None

    try:
        city_results = _geocode_results(destination) or _geocode_results(city_part)
        if not city_results:
            errors.append(
                f"destination '{destination}' could not be found — check the spelling"
            )
            return

        if not stated_country:
            return

        city_country_codes = {
            r["country_code"] for r in city_results if r["country_code"]
        }

        country_results = _geocode_results(stated_country, count=1)
        if not country_results:
            # Can't resolve the stated country at all — nothing to compare
            # against, so don't block the request over it.
            return

        stated_country_code = country_results[0]["country_code"]
        if stated_country_code and stated_country_code not in city_country_codes:
            actual_countries = sorted({
                r["country"] for r in city_results if r["country"]
            })
            errors.append(
                f"'{city_part}' was found, but not in '{stated_country}'. "
                f"'{city_part}' actually exists in: "
                f"{', '.join(actual_countries) or 'a different country'}"
            )
    except GeocodingRateLimited:
        # We got rate-limited, NOT a "place not found" signal — skip the
        # check silently rather than rejecting a valid destination because
        # of our own request volume.
        pass
    except requests.RequestException:
        # Geocoding service unreachable/timed out — skip this check rather
        # than blocking legitimate requests on a third-party outage.
        pass


def validate_travel_request(req: dict) -> list[str]:
    """Run all Tier 1 + Tier 2 checks and return a list of human-readable
    error messages (empty list means the request is valid)."""
    errors: list[str] = []
    _validate_dates(req, errors)
    _validate_travelers(req, errors)
    _validate_budget(req, errors)
    _validate_destination(req, errors)
    return errors