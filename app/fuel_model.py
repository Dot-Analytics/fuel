"""
fuel_model.py

Refactored from DTI_FUEL_POC.ipynb
- No Streamlit
- Pure Python
- Callable from API / job / SPCS service
"""

import os
import sys
import json
import math
import time
import heapq
import re
import urllib.parse
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import requests
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
# Core fuel / routing logic (THIS IS THE NOTEBOOK LOGIC)
# ---------------------------------------------------------------------

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


def score_lane(row: pd.Series) -> float:
    """
    Scoring heuristic from notebook.
    """
    score = 0.0

    score += row.get("fuel_cost", 0.0)
    score += row.get("toll_cost", 0.0)

    if row.get("is_cross_border"):
        score *= 1.15

    return score


def select_best_lanes(lanes_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    lanes_df = lanes_df.copy()
    lanes_df["score"] = lanes_df.apply(score_lane, axis=1)

    return lanes_df.sort_values("score").head(top_n)


def lanes_to_records(lanes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    sanitized = lanes_df.replace([np.nan, np.inf, -np.inf], None)
    return sanitized.to_dict(orient="records")


# ---------------------------------------------------------------------
# Main public entrypoint
# ---------------------------------------------------------------------

def run_fuel_model(
    run_date: str,
    scenario: str = "default",
    top_n: int = 5
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

    # 3. Select best lanes
    best_lanes = select_best_lanes(lanes_df, top_n=top_n)

    # 4. Optional: LLM explanation
    explanation = None
    if os.getenv("ENABLE_LLM_EXPLANATION", "false").lower() == "true":
        prompt = f"""
        Explain why these lanes were selected as optimal:
        {best_lanes.to_dict(orient='records')}
        """
        explanation = call_openai(prompt)

    elapsed = round(time.time() - start_ts, 2)

    return {
        "run_date": run_date,
        "scenario": scenario,
        "elapsed_seconds": elapsed,
        "lane_count": len(best_lanes),
        "lanes": lanes_to_records(best_lanes),
        "explanation": explanation,
    }
