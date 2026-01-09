"""
fuel_model.py

Refactored from DTI_FUEL_POC.ipynb
- No Streamlit
- Pure Python
- Callable from API / job / SPCS service
"""

import os
import json
import math
import time
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import openai
import snowflake.connector


# ---------------------------------------------------------------------
# Constants (from notebook)
# ---------------------------------------------------------------------

DTI_DEFAULT_CLOSURE_ID = "2104"
DEFAULT_PROFILE_ID = "15073630"

METERS_PER_MILE = 1609.344

OPENAI_MODEL = "gpt-5-nano"
MAX_TOKENS = 10_000

STATE_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}


# ---------------------------------------------------------------------
# Snowflake connection (SPCS-compatible OAuth)
# ---------------------------------------------------------------------

TOKEN_PATH = "/snowflake/session/token"


def get_snowflake_connection():
    with open(TOKEN_PATH) as f:
        token = f.read().strip()

    return snowflake.connector.connect(
        host=os.getenv("SNOWFLAKE_HOST"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        authenticator="oauth",
        token=token,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


# ---------------------------------------------------------------------
# Helper functions (lifted directly from notebook)
# ---------------------------------------------------------------------

def meters_to_miles(meters: float) -> float:
    return meters / METERS_PER_MILE


def normalize_state(state: str) -> str:
    if not state:
        return None
    return STATE_TO_ABBR.get(state.strip(), state.strip())


def call_openai(prompt: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )

    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------
# Core fuel / routing logic (based on notebook concepts)
# ---------------------------------------------------------------------

GALLON_STEP = float(os.getenv("FUEL_GALLON_STEP", "1.0"))


def _parse_json_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.7613
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_miles * math.asin(math.sqrt(a))


def _extract_point_fields(point: dict) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[float]]:
    lon = point.get("lon") or point.get("LON") or point.get("longitude")
    lat = point.get("lat") or point.get("LAT") or point.get("latitude")
    state = point.get("state") or point.get("STATE") or point.get("state_abbr")
    mm = point.get("mm") or point.get("MM") or point.get("mile") or point.get("MM_APPROX_MI")
    try:
        lon = float(lon) if lon is not None else None
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lon = None
        lat = None
    try:
        mm = float(mm) if mm is not None else None
    except (TypeError, ValueError):
        mm = None
    return lon, lat, normalize_state(state) if state else None, mm


def parse_geotunnel_points(value: Any) -> List[Dict[str, Any]]:
    points = _parse_json_maybe(value)
    if not points:
        return []
    parsed = []
    for item in points:
        if isinstance(item, dict):
            lon, lat, state, mm = _extract_point_fields(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lon, lat = item[0], item[1]
            try:
                lon = float(lon)
                lat = float(lat)
            except (TypeError, ValueError):
                continue
            state = None
            mm = None
        else:
            continue
        parsed.append({"lon": lon, "lat": lat, "state": state, "mm": mm})
    return parsed


def build_state_events_from_geotunnel(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not points:
        return []
    events = []
    cumulative_miles = 0.0
    prev = None
    last_state = None
    for idx, point in enumerate(points):
        lon = point.get("lon")
        lat = point.get("lat")
        state = point.get("state")
        mm = point.get("mm")
        if prev and prev.get("lat") is not None and prev.get("lon") is not None and lat is not None and lon is not None:
            if mm is None or prev.get("mm") is None:
                cumulative_miles += _haversine_miles(prev["lat"], prev["lon"], lat, lon)
            else:
                cumulative_miles = max(cumulative_miles, float(mm))
        if state and state != last_state:
            events.append(
                {
                    "event_seq": len(events) + 1,
                    "mile_marker": cumulative_miles,
                    "state": state,
                    "event_type": "STATE",
                    "location_type": "STATE",
                }
            )
            last_state = state
        if idx == 0 and state and not events:
            events.append(
                {
                    "event_seq": 1,
                    "mile_marker": 0.0,
                    "state": state,
                    "event_type": "STATE",
                    "location_type": "STATE",
                }
            )
            last_state = state
        prev = point
    last_mile_marker = cumulative_miles
    if last_state is not None:
        if not events or events[-1]["mile_marker"] < last_mile_marker:
            events.append(
                {
                    "event_seq": len(events) + 1,
                    "mile_marker": last_mile_marker,
                    "state": last_state,
                    "event_type": "END",
                    "location_type": "END",
                    "can_buy": False,
                }
            )
    return events


def _detect_state_price_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return None, None
    cols = list(df.columns)
    cols_u = [c.upper() for c in cols]
    state_candidates = ["STATE", "STATE_ABBR", "STATE_CODE", "ST", "STATEPROV"]
    price_candidates = ["RCSP", "PRICE", "PPG", "FUEL_PRICE", "DIESEL_PRICE", "STATE_PRICE"]
    state_col = next((cols[cols_u.index(c)] for c in state_candidates if c in cols_u), None)
    price_col = next((cols[cols_u.index(c)] for c in price_candidates if c in cols_u), None)
    return state_col, price_col


def build_state_price_map(fuel_costs_df: pd.DataFrame) -> Dict[str, float]:
    state_col, price_col = _detect_state_price_columns(fuel_costs_df)
    if not state_col or not price_col:
        return {}
    df = fuel_costs_df[[state_col, price_col]].copy()
    df[state_col] = df[state_col].apply(normalize_state)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[state_col, price_col])
    return df.groupby(state_col)[price_col].min().to_dict()


def _parse_stop_events(value: Any) -> List[Dict[str, Any]]:
    stops = _parse_json_maybe(value)
    if not stops:
        return []
    out = []
    for item in stops:
        if not isinstance(item, dict):
            continue
        mm = item.get("mile_marker") or item.get("mm") or item.get("MM_APPROX_MI")
        event_type = str(item.get("event_type") or item.get("type") or "STOP").upper()
        location_type = "STOP"
        if event_type in {"DC", "DISTRIBUTION_CENTER"}:
            location_type = "DC"
            event_type = "DC"
        elif event_type in {"HOS", "BREAK", "REST"}:
            location_type = "HOS"
            event_type = "STOP"
        try:
            mm = float(mm)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "mile_marker": mm,
                "event_type": event_type,
                "location_type": location_type,
                "state": normalize_state(item.get("state") or item.get("STATE")),
                "is_expected_stop": bool(item.get("is_expected_stop") or event_type in {"HOS", "BREAK", "REST"}),
            }
        )
    return out


def merge_events(state_events: List[Dict[str, Any]], stop_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = state_events + stop_events
    if not events:
        return []
    events.sort(key=lambda r: (r.get("mile_marker") or 0.0, r.get("event_type") or ""))
    deduped = []
    seen = set()
    for event in events:
        key = (round(float(event.get("mile_marker") or 0.0), 3), event.get("event_type"))
        if key in seen:
            continue
        seen.add(key)
        event["event_seq"] = len(deduped) + 1
        event.setdefault("can_buy", True)
        deduped.append(event)
    for idx, event in enumerate(deduped):
        if idx < len(deduped) - 1:
            event["miles_to_next"] = max(0.0, float(deduped[idx + 1]["mile_marker"]) - float(event["mile_marker"]))
        else:
            event["miles_to_next"] = 0.0
    return deduped


def optimize_fuel_plan_rcsp(
    events: List[Dict[str, Any]],
    initial_fuel: float,
    mpg: float,
    tank_capacity: float,
    safety_buffer: float,
    stop_fee: float,
) -> Tuple[List[Tuple[int, float]], float]:
    if not events:
        return [], 0.0
    mpg = max(mpg, 0.1)
    step = max(GALLON_STEP, 0.1)
    capacity = max(tank_capacity, step)
    levels = np.arange(0.0, capacity + step / 2, step)
    level_count = len(levels)
    needed = []
    for idx, event in enumerate(events):
        gallons_needed = float(event.get("miles_to_next") or 0.0) / mpg
        gallons_needed = math.ceil(gallons_needed / step) * step
        buffer = safety_buffer if idx < len(events) - 1 else 0.0
        needed.append(gallons_needed + buffer)

    inf = float("inf")
    costs = np.full(level_count, inf)
    prev = [dict() for _ in range(len(events))]

    start_fuel = min(max(initial_fuel, 0.0), capacity)
    start_idx = int(round(start_fuel / step))
    costs[start_idx] = 0.0

    for idx, event in enumerate(events):
        price = float(event.get("price") or 0.0)
        location_type = str(event.get("location_type") or "").upper()
        is_expected_stop = bool(event.get("is_expected_stop"))
        fee_applies = location_type not in {"DC", "HOS"} and not is_expected_stop and stop_fee > 0
        can_buy = bool(event.get("can_buy", True))
        next_costs = np.full(level_count, inf)
        next_prev = {}
        for i, current_cost in enumerate(costs):
            if current_cost == inf:
                continue
            current_fuel = levels[i]
            for j in range(i, level_count):
                target_fuel = levels[j]
                buy_gallons = target_fuel - current_fuel
                if buy_gallons > 0 and not can_buy:
                    continue
                extra_cost = buy_gallons * price
                if buy_gallons > 0 and fee_applies:
                    extra_cost += stop_fee
                total_cost = current_cost + extra_cost
                remaining = target_fuel - needed[idx]
                if remaining < 0:
                    continue
                remaining = max(0.0, remaining)
                remaining_idx = int(round(remaining / step))
                if total_cost < next_costs[remaining_idx]:
                    next_costs[remaining_idx] = total_cost
                    next_prev[remaining_idx] = (i, buy_gallons)
        costs = next_costs
        prev[idx] = next_prev

    end_idx = int(np.argmin(costs))
    min_cost = float(costs[end_idx]) if costs[end_idx] != inf else 0.0

    plan = []
    cur_idx = end_idx
    for idx in reversed(range(len(events))):
        step_prev = prev[idx].get(cur_idx)
        if not step_prev:
            break
        prev_idx, buy_gallons = step_prev
        if buy_gallons > 0:
            plan.append((events[idx]["event_seq"], round(float(buy_gallons), 2)))
        cur_idx = prev_idx
    plan.reverse()
    return plan, min_cost

def load_base_data(conn) -> Dict[str, pd.DataFrame]:
    """
    Loads all base Snowflake datasets used by the model.
    """

    datasets = {}

    queries = {
        "lanes": """
            SELECT *
            FROM ANALYTICS.DTI_FUEL_LANES
        """,
        "facilities": """
            SELECT *
            FROM ANALYTICS.DTI_FACILITIES
        """,
        "fuel_costs": """
            SELECT *
            FROM ANALYTICS.DTI_FUEL_COSTS
        """
    }

    for name, sql in queries.items():
        df = pd.read_sql(sql, conn)
        datasets[name] = df

    return datasets


def compute_lane_distance_costs(lanes_df: pd.DataFrame) -> pd.DataFrame:
    lanes_df = lanes_df.copy()

    lanes_df["distance_miles"] = lanes_df["distance_meters"].apply(meters_to_miles)
    lanes_df["fuel_cost"] = lanes_df["distance_miles"] * lanes_df["fuel_rate"]

    return lanes_df


def lanes_to_records(lanes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    sanitized = lanes_df.replace([np.nan, np.inf, -np.inf], None)
    return sanitized.to_dict(orient="records")


# ---------------------------------------------------------------------
# Main public entrypoint
# ---------------------------------------------------------------------

def run_fuel_model(
    run_date: str,
    scenario: str = "default",
    top_n: int = 5,
    mpg: float = 6.5,
    tank_capacity: float = 200.0,
    initial_fuel: float = 100.0,
    safety_buffer: float = 10.0,
    stop_fee: float = 100.0,
) -> Dict[str, Any]:
    """
    Main callable entrypoint.

    This is what your API / job / SPCS service should call.
    """

    start_ts = time.time()

    conn = get_snowflake_connection()

    # 1. Load data
    datasets = load_base_data(conn)
    lanes_df = datasets["lanes"]

    # 2. Compute costs
    lanes_df = compute_lane_distance_costs(lanes_df)

    # 3. Build state price map (RCSP-style lowest price per state)
    fuel_costs_df = datasets.get("fuel_costs")
    state_price_map = build_state_price_map(fuel_costs_df)

    plan_rows = []
    for _, lane in lanes_df.iterrows():
        points = parse_geotunnel_points(lane.get("geotunnel_points") or lane.get("GEOTUNNEL_POINTS"))
        state_events = build_state_events_from_geotunnel(points)
        if not state_events:
            origin_state = normalize_state(lane.get("origin_state") or lane.get("ORIGIN_STATE"))
            dest_state = normalize_state(lane.get("dest_state") or lane.get("DEST_STATE"))
            distance_miles = float(lane.get("distance_miles") or 0.0)
            state_events = [
                {
                    "event_seq": 1,
                    "mile_marker": 0.0,
                    "state": origin_state,
                    "event_type": "STATE",
                    "location_type": "STATE",
                },
                {
                    "event_seq": 2,
                    "mile_marker": distance_miles,
                    "state": dest_state or origin_state,
                    "event_type": "END",
                    "location_type": "END",
                    "can_buy": False,
                },
            ]

        stop_events = _parse_stop_events(lane.get("stop_events") or lane.get("STOP_EVENTS"))
        events = merge_events(state_events, stop_events)

        for event in events:
            state = normalize_state(event.get("state"))
            event["state"] = state
            event_price = state_price_map.get(state)
            if event.get("location_type") == "DC" and lane.get("dc_price") is not None:
                event_price = lane.get("dc_price")
            if event_price is None:
                event_price = lane.get("fuel_rate") or lane.get("FUEL_RATE")
            event["price"] = float(event_price) if event_price is not None else 0.0
            if not event.get("can_buy", True):
                event["price"] = 0.0
            event["gallons_to_next"] = (float(event.get("miles_to_next") or 0.0) / max(mpg, 0.1))

        plan, plan_cost = optimize_fuel_plan_rcsp(
            events,
            initial_fuel=initial_fuel,
            mpg=mpg,
            tank_capacity=tank_capacity,
            safety_buffer=safety_buffer,
            stop_fee=stop_fee,
        )

        plan_rows.append(
            {
                "lane": lane.to_dict(),
                "events": events,
                "fuel_plan": [{"event_seq": seq, "buy_gallons": gallons} for seq, gallons in plan],
                "plan_cost": round(plan_cost, 2),
            }
        )

    plan_rows.sort(key=lambda row: row.get("plan_cost") or 0.0)
    best_plans = plan_rows[:top_n]

    # 4. Optional: LLM explanation
    explanation = None
    if os.getenv("ENABLE_LLM_EXPLANATION", "false").lower() == "true":
        prompt = f"""
        Explain why these fuel plans were selected as optimal:
        {best_plans}
        """
        explanation = call_openai(prompt)

    elapsed = round(time.time() - start_ts, 2)

    return {
        "run_date": run_date,
        "scenario": scenario,
        "elapsed_seconds": elapsed,
        "lane_count": len(best_plans),
        "lanes": best_plans,
        "explanation": explanation,
    }
