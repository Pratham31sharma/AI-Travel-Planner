from langchain_core.tools import tool

@tool
def generate_packing_list(climate: str, interests: str, trip_length_days: int) -> str:
    """Generate a packing list given a rough climate description
    (e.g. 'hot and humid', 'cold and snowy'), a comma-separated list of
    trip interests (e.g. 'hiking, museums, beach'), and trip length in days.
    """
    base = ["Passport / ID", "Phone charger", "Toiletries", "Reusable water bottle",
            "Basic first-aid kit", "Travel adapter"]

    climate = climate.lower()
    if any(w in climate for w in ["hot", "humid", "tropical", "warm"]):
        base += ["Lightweight breathable clothing", "Sunscreen", "Sunglasses", "Hat"]
    if any(w in climate for w in ["cold", "snow", "winter", "chilly"]):
        base += ["Thermal layers", "Waterproof jacket", "Gloves", "Warm hat"]
    if any(w in climate for w in ["rain", "wet", "monsoon"]):
        base += ["Compact umbrella", "Waterproof shoes"]

    interest_list = [i.strip().lower() for i in interests.split(",") if i.strip()]
    if any("hik" in i or "trek" in i for i in interest_list):
        base += ["Hiking boots", "Daypack", "Blister plasters"]
    if any("beach" in i or "swim" in i for i in interest_list):
        base += ["Swimwear", "Beach towel", "Flip-flops"]
    if any("museum" in i or "culture" in i or "temple" in i for i in interest_list):
        base += ["Modest clothing for religious/cultural sites"]
    if any("business" in i or "conference" in i for i in interest_list):
        base += ["Formal outfit", "Laptop + charger"]

    clothing_note = f"~{max(trip_length_days // 2, 2)} outfit changes (laundry available at most stays)"
    base.append(clothing_note)

    return "\n".join(f"- {item}" for item in base)
