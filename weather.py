import requests
from langchain_core.tools import tool

@tool
def get_weather_outlook(city: str, start_date: str, end_date: str) -> str:
    """Get a weather outlook for a city between two dates (YYYY-MM-DD).
    Uses geocoding + a forecast/climate API. Useful for packing and
    activity-planning decisions. Falls back to seasonal averages if the
    dates are too far in the future for a daily forecast.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"[weather: could not geocode '{city}']"
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_mean",
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "auto",
            },
            timeout=10,
        ).json()

        daily = forecast.get("daily")
        if not daily or not daily.get("time"):
            return (
                f"[weather: no daily forecast available for {city} on those dates "
                "— likely outside the ~16 day forecast window. Consider seasonal "
                "norms for that time of year instead.]"
            )

        lines = [f"Weather outlook for {city}:"]
        for i, day in enumerate(daily["time"]):
            hi = daily["temperature_2m_max"][i]
            lo = daily["temperature_2m_min"][i]
            rain = daily["precipitation_probability_mean"][i]
            lines.append(f"- {day}: {lo}-{hi}°C, {rain}% chance of rain")
        return "\n".join(lines)

    except requests.RequestException as e:
        return f"[weather error: {e}]"
