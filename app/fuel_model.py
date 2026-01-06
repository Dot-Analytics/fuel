# ---- cell 0 ----
import json
import math
import os
import sys
import time
import uuid
import heapq
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
import openai
import pandas as pd
import requests
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, udf
from snowflake.snowpark.window import Window
from snowflake.snowpark.types import (
    DecimalType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


TOKEN_PATH = "/snowflake/session/token"


def _build_spcs_session() -> Session:
    with open(TOKEN_PATH) as f:
        token = f.read().strip()

    configs = {
        "authenticator": "oauth",
        "token": token,
        "host": os.getenv("SNOWFLAKE_HOST"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }

    return Session.builder.configs(configs).create()


try:
    session = get_active_session()
except Exception:
    session = _build_spcs_session()


# ---- cell 1 ----
INTEGRATION_NAME = 'DTI_FUEL'

PCMILER_API_KEY = os.environ["PCMILER_API_KEY"]
PC_MILER_API_KEY = PCMILER_API_KEY
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SAMSARA_API_KEY = os.environ["SAMSARA_API_KEY"]

ROUTE_REPORTS_BASE_URL = 'https://pcmiler.alk.com/apis/rest/v1.0/Service.svc/route/routeReports'
AVOIDFAVOR_BASE_URL     = 'https://fleets.trimblemaps.com/api/assets/v1/avoidFavorSets'
VEHICLE_GROUP_AF_URL     = AVOIDFAVOR_BASE_URL + '/vehicleGroups'

DTI_VEHICLE_GROUP_ID    = '1543010'
DTI_DEFAULT_CLOSURE_ID  = '2104'
DEFAULT_PROFILE_ID      = '15073630'

METERS_PER_MILE = 1609.344

OPENAI_MODEL = 'gpt-5-nano'
MAX_TOKENS = 10000

STATE_TO_ABBR = {
        "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
        "Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
        "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
        "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
        "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
        "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
        "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
        "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
        "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC",
    }

BASE = "https://api.samsara.com"
HEADERS = {"Authorization": f"Bearer {SAMSARA_API_KEY}"}

# ---- cell 2 ----
DEBUG_PCM = os.environ.get("DEBUG_PCM", "0") in ("1", "true", "True")

def _debug(msg):
    if DEBUG_PCM:
        print(f"[PCM-DEBUG] {msg}")


def send_error_email(body, subject='DTI FUEL ERROR', recipients='joe.haenel@dotfoods.com'):
    session.call(
        "SYSTEM$SEND_EMAIL", # Stored procedure name
        INTEGRATION_NAME,
        recipients,
        subject, # email_subject
        body # email_content
    )
    return True


# ---- cell 3 (moved) ----
try:
    # Load in Wex Fuel data summary
    DTI_FUEL_DATA_SUMMARY_SP = session.table("DEV_DOT_MLAI.DTI_FUEL.WEX_FUEL_DATA_SUMMARY")

    # Load in DC Fuel Data
    DC_FUEL_DATA_SP = session.table("DEV_DOT_MLAI.DTI_FUEL.DTI_HOMEFUEL_CURRENT_PRICE")

except Exception as error:
    print(f"loading data failed: {error}")
    send_error_email(body=f'{error}')


# ---- cell 5 ----
def sample_geotunnel(points, step_mi: float = 1.0):
    """Resample a polyline [(lon,lat),...] into ~1-mile spaced points with mile markers."""
    out = []
    if not points: 
        return out
    from math import radians, sin, cos, asin, sqrt
    def hav_mi(lat1, lon1, lat2, lon2):
        R = 3958.7613
        dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 2*R*asin(sqrt(a))

    acc = 0.0
    lon0, lat0 = points[0]
    out.append((lon0, lat0, 0.0))
    carry = 0.0
    for (lon1, lat1) in points[1:]:
        seg = hav_mi(lat0, lon0, lat1, lon1)
        while carry + seg >= step_mi and seg > 0:
            t = (step_mi - carry) / seg
            lon = lon0 + t*(lon1-lon0)
            lat = lat0 + t*(lat1-lat0)
            acc += step_mi
            out.append((lon, lat, acc))
            lon0, lat0 = lon, lat
            seg -= (step_mi - carry)
            carry = 0.0
        carry += seg
        lon0, lat0 = lon1, lat1
    return out

# ---- cell 6 ----
def is_team_driving(stops_sp) -> bool:
    """
    Team = both driver numbers present and non-zero on the first stop.
    Works with '', None, or '0' coming from Snowflake.
    """
    first = (
        stops_sp.sort(F.col("LOAD_STOP_SEQUENCE_NO").asc())
                .select("DRIVER_ONE_EMP_NO", "DRIVER_TWO_EMP_NO")
                .limit(1)
                .to_pandas()
    )
    if first.empty:
        return False
    d1 = str(first.iloc[0].get("DRIVER_ONE_EMP_NO") or "").strip()
    d2 = str(first.iloc[0].get("DRIVER_TWO_EMP_NO") or "").strip()
    def _has(val: str) -> bool:
        try:
            # treat numeric > 0 as present; ignore blanks/None/'0'
            return int(float(val)) > 0
        except Exception:
            return len(val) > 0
    return _has(d1) and _has(d2)


# ---- cell 7 ----
def expand_dc_candidates_from_geotunnel(
    session,
    routes_with_geotunnel_pd,   # pandas DF from build_routes_df_for_events(...)
    *,
    radius_mi: float = 5.0,
    step_mi: float = 1.0,
    dot_table: str = "DEV_DOT_MLAI.DTI_FUEL.DTI_DOT_LOCATIONS",
) -> "pd.DataFrame":
    """
    For each leg, sample the geotunnel, find nearby DOTs within `radius_mi`,
    dedupe to the nearest per (LOAD_NO, FROM_SEQ, TO_SEQ, DOT_NAME),
    and return DC candidate events with haversine distance (miles).

    Debug logging:
      - Set env DTI_FUEL_LOG_LEVEL (e.g., DEBUG, INFO) to control verbosity.
      - Set env DTI_FUEL_EXPLAIN=1 to print query plans for logged steps.
      - Set env DTI_FUEL_LOG_MAX_LEGS (default 5) to limit per-leg logging.
      - We also set a session QUERY_TAG to correlate queries in Snowsight.

    Tunables / guardrails (env):
      - DTI_FUEL_MAX_SAMPLES (default 600): cap per-leg geotunnel samples.
      - DTI_FUEL_MAX_CANDIDATES (default 5_000_000): skip legs with exploding
        bbox-join candidate counts.

    Requires utilities in this module:
      - sample_geotunnel(points, step_mi)
      - _polyline_length_mi(points)
    """
    import os, uuid, time, traceback, math
    import pandas as pd
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.window import Window  # kept for compatibility (not used now)

    # --------------------------- logging helpers ----------------------------
    import logging
    LOG_LEVEL = os.getenv("DTI_FUEL_LOG_LEVEL", "INFO").upper()
    EXPLAIN   = os.getenv("DTI_FUEL_EXPLAIN", "0") == "1"
    LOG_MAX_LEGS = int(os.getenv("DTI_FUEL_LOG_MAX_LEGS", "5"))

    MAX_SAMPLES = int(os.getenv("DTI_FUEL_MAX_SAMPLES", "600"))
    MAX_CANDIDATES = int(os.getenv("DTI_FUEL_MAX_CANDIDATES", "5000000"))

    logger = logging.getLogger("dti_fuel.expand_dc_candidates_from_geotunnel")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    run_id = str(uuid.uuid4())[:8]
    try:
        session.sql(f"alter session set query_tag = 'dti_fuel:expand_dc:{run_id}'").collect()
    except Exception:
        # Non-fatal: logging only
        logger.debug(f"[{run_id}] Could not set query_tag: {traceback.format_exc()}")

    def _last_qid():
        try:
            return session.sql("select last_query_id()").collect()[0][0]
        except Exception:
            return None

    def log_shape(df, label: str, do_explain: bool = False):
        """Executes COUNT to materialize, logs row count + last query id; optional EXPLAIN."""
        try:
            t0 = time.perf_counter()
            n = df.count()
            qid = _last_qid()
            dt = (time.perf_counter() - t0) * 1000.0
            logger.info(f"[{run_id}] {label}: rows={n} ({dt:.1f} ms){' qid='+qid if qid else ''}")
            if do_explain:
                logger.debug(f"[{run_id}] {label}: EXPLAIN start")
                df.explain()  # prints to stdout; captured by Snowflake logs
                logger.debug(f"[{run_id}] {label}: EXPLAIN end")
            return n
        except Exception:
            logger.error(f"[{run_id}] {label}: count/explain failed\n{traceback.format_exc()}")
            return None

    METERS_PER_MILE = 1609.344
    radius_m = float(radius_mi) * METERS_PER_MILE

    # ---- DOT table prep (once) ----------------------------------------------
    try:
        d_raw = session.table(dot_table)
        d_cols = [f.name for f in d_raw.schema.fields]
        d_cols_u = [c.upper() for c in d_cols]
        logger.info(f"[{run_id}] DOT table '{dot_table}' columns={d_cols_u}")
    except Exception:
        logger.error(f"[{run_id}] Failed to open DOT table {dot_table}\n{traceback.format_exc()}")
        raise

    # Resolve LAT/LON (support LONGITUDE/LATITUDE fallbacks)
    def _pick(df, cols_u, *names):
        for n in names:
            if n in cols_u:
                return d_cols[cols_u.index(n)]
        return None

    d = d_raw
    lon_name = _pick(d, d_cols_u, "LON", "LONGITUDE")
    lat_name = _pick(d, d_cols_u, "LAT", "LATITUDE")
    if lon_name is None or lat_name is None:
        msg = f"{dot_table} must contain LON/LAT (or LONGITUDE/LATITUDE). Got columns: {d_cols_u}"
        logger.error(f"[{run_id}] {msg}")
        raise ValueError(msg)

    if "LON" not in d_cols_u:
        d = d.with_column("LON", d[lon_name])
    if "LAT" not in d_cols_u:
        d = d.with_column("LAT", d[lat_name])

    # Guardrails (numeric only; avoid GEOGRAPHY entirely)
    d = d.filter(
        F.col("LAT").is_not_null() & F.col("LON").is_not_null() &
        F.col("LAT").between(-90, 90) & F.col("LON").between(-180, 180)
    )

    # Add STATE if present (no polygon join fallback to avoid GEOGRAPHY ops)
    d_cols = [f.name for f in d.schema.fields]
    d_cols_u = [c.upper() for c in d_cols]
    dot_state_col_u = next(
        (c for c in ["STATE", "STATE_ABBR", "STATE_ABBRV", "STATE_CODE", "ST", "STATEPROV"] if c in d_cols_u),
        None
    )
    if dot_state_col_u:
        d_aug = d.with_column("STATE", F.upper(d[dot_state_col_u]))
    else:
        d_aug = d.with_column("STATE", F.lit(None))

    # Ensure DOT_NAME exists; map common alternatives if needed
    d_aug_cols   = [f.name for f in d_aug.schema.fields]
    d_aug_cols_u = [c.upper() for c in d_aug_cols]
    if "DOT_NAME" not in d_aug_cols_u:
        alt_name_u = next(
            (c for c in ["DOT", "DOTNAME", "NAME", "LOCATION_NAME", "DOT_LOCATION_NAME"] if c in d_aug_cols_u),
            None
        )
        if not alt_name_u:
            msg = f"{dot_table} must contain DOT_NAME (or DOT/DOTNAME/NAME/LOCATION_NAME). Columns: {d_aug_cols_u}"
            logger.error(f"[{run_id}] {msg}")
            raise ValueError(msg)
        d_aug = d_aug.with_column("DOT_NAME", d_aug[alt_name_u])

    # Slim to only needed columns BEFORE any joins (no GEOGRAPHY columns)
    d_aug = d_aug.select("DOT_NAME", "STATE", "LON", "LAT")
    log_shape(d_aug, "d_aug (DOT catalog)", do_explain=EXPLAIN)

    # ---- Per-leg processing ---------------------------------------------------
    out_chunks = []
    leg_idx = 0

    for r in routes_with_geotunnel_pd.itertuples(index=False):
        leg_idx += 1
        log_this_leg = leg_idx <= LOG_MAX_LEGS  # limit verbose logging

        try:
            load_no = int(getattr(r, "LOAD_NO"))
            from_seq = int(getattr(r, "LOAD_STOP_SEQUENCE_NO"))
            to_seq   = from_seq + 1
            pts      = getattr(r, "GEOTUNNEL_POINTS", None)
        except Exception:
            logger.error(f"[{run_id}] Leg[{leg_idx}] row parse failed\n{traceback.format_exc()}")
            continue

        if not pts:
            if log_this_leg:
                logger.info(f"[{run_id}] Leg[{leg_idx}] load={load_no} seq={from_seq}->{to_seq}: no geotunnel points; skipping")
            continue

        # per-leg mile-marker scaling
        leg_pcm = getattr(r, "LEG_MI_PCM", None)
        scale   = getattr(r, "LEG_SCALE", None)
        if scale is None:
            try:
                gt_len = _polyline_length_mi(pts)
                scale  = (float(leg_pcm)/gt_len) if (leg_pcm and gt_len) else 1.0
            except Exception:
                logger.error(f"[{run_id}] Leg[{leg_idx}] scale compute failed\n{traceback.format_exc()}")
                scale = 1.0

        # avoid hopping over nearby DCs
        step = min(float(step_mi), max(0.25, float(radius_mi)/2.0))
        try:
            samples = sample_geotunnel(pts, step_mi=step)
        except Exception:
            logger.error(f"[{run_id}] Leg[{leg_idx}] sample_geotunnel failed\n{traceback.format_exc()}")
            continue

        if not samples:
            if log_this_leg:
                logger.info(f"[{run_id}] Leg[{leg_idx}] load={load_no} seq={from_seq}->{to_seq}: 0 samples after step={step}")
            continue

        # ---- Cap samples on very long legs (guardrail) ----------------------
        if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
            stride = math.ceil(len(samples) / MAX_SAMPLES)
            samples = samples[::stride]
            if log_this_leg:
                logger.info(f"[{run_id}] Leg[{leg_idx}] capped samples to {len(samples)} (stride={stride})")

        # ---- Compute a per-leg bounding box & prefilter DOTs ----------------
        leg_lats = [lat for (_, lat, _) in samples]
        leg_lons = [lon for (lon, _, _) in samples]
        mid_lat  = sum(leg_lats) / len(leg_lats)

        pad_deg_lat = (radius_m / 111_320.0) * 1.5  # 1.5x safety pad
        pad_deg_lon = pad_deg_lat / max(1e-6, math.cos(math.radians(mid_lat)))

        min_lat, max_lat = min(leg_lats)-pad_deg_lat, max(leg_lats)+pad_deg_lat
        min_lon, max_lon = min(leg_lons)-pad_deg_lon, max(leg_lons)+pad_deg_lon

        d_leg = d_aug.filter(
            (F.col("LAT").between(min_lat, max_lat)) &
            (F.col("LON").between(min_lon, max_lon))
        )
        if log_this_leg:
            log_shape(d_leg, f"Leg[{leg_idx}] d_leg (prefiltered DOTs)", do_explain=False)

        # ---- Build tiny per-leg DataFrame (NO GEOGRAPHY) --------------------
        leg_rows = [
            (load_no, from_seq, to_seq, idx, float(lon), float(lat), float(mm) * float(scale))
            for idx, (lon, lat, mm) in enumerate(samples)
        ]
        pts_df = session.create_dataframe(
            leg_rows, schema=["LOAD_NO","FROM_SEQ","TO_SEQ","PT_IDX","LON","LAT","MM"]
        ).filter(
            F.col("LAT").is_not_null() & F.col("LON").is_not_null()
        )

        if log_this_leg:
            logger.info(f"[{run_id}] Leg[{leg_idx}] load={load_no} {from_seq}->{to_seq}: samples={len(leg_rows)}")
            log_shape(pts_df, f"Leg[{leg_idx}] pts_df", do_explain=False)

        # ---------- Numeric bounding box join (per-leg reduced) --------------
        p  = pts_df.alias("P")
        d2 = d_leg.alias("D")

        lat_delta_deg = F.lit(radius_m / 111320.0)  # ~meters per degree latitude
        lon_den       = F.greatest(F.lit(1e-6), F.call_function("COS", F.call_function("RADIANS", p["LAT"])))
        lon_delta_deg = F.lit(radius_m / 111320.0) / lon_den

        cond_bb = (F.abs(d2["LAT"] - p["LAT"]) <= lat_delta_deg) & (F.abs(d2["LON"] - p["LON"]) <= lon_delta_deg)
        cand_bb = p.join(d2, cond_bb, how="inner")
        cand_cnt = log_shape(cand_bb, f"Leg[{leg_idx}] cand_bb (bbox join)", do_explain=EXPLAIN)

        # Guardrail: skip legs that still explode
        if cand_cnt is not None and MAX_CANDIDATES and cand_cnt > MAX_CANDIDATES:
            logger.warning(f"[{run_id}] Leg[{leg_idx}] too many candidates ({cand_cnt} > {MAX_CANDIDATES}); skipping leg")
            continue

        # ---------- Core candidate columns (still no GEOGRAPHY) -------------
        cand_core = cand_bb.select(
            p["LOAD_NO"].alias("LOAD_NO"),
            p["FROM_SEQ"].alias("FROM_SEQ"),
            p["TO_SEQ"].alias("TO_SEQ"),
            d2["DOT_NAME"].alias("DOT_NAME"),
            d2["STATE"].alias("STATE"),
            p["MM"].alias("MM"),
            p["LAT"].alias("PT_LAT"),   p["LON"].alias("PT_LON"),
            d2["LAT"].alias("DOT_LAT"), d2["LON"].alias("DOT_LON"),
        )
        if log_this_leg:
            log_shape(cand_core, f"Leg[{leg_idx}] cand_core", do_explain=False)

        # ---------- Haversine approx for distance (meters) -------------------
        R = 6371008.8  # meters
        dlat = F.call_function("RADIANS", F.col("DOT_LAT") - F.col("PT_LAT"))
        dlon = F.call_function("RADIANS", F.col("DOT_LON") - F.col("PT_LON"))
        phi1 = F.call_function("RADIANS", F.col("PT_LAT"))
        phi2 = F.call_function("RADIANS", F.col("DOT_LAT"))
        sin_dlat = F.call_function("SIN", dlat / F.lit(2.0))
        sin_dlon = F.call_function("SIN", dlon / F.lit(2.0))
        a = sin_dlat * sin_dlat + F.call_function("COS", phi1) * F.call_function("COS", phi2) * sin_dlon * sin_dlon
        c = F.call_function("ASIN", F.call_function("SQRT", a)) * F.lit(2.0)
        cand_hav = cand_core.with_column("D_APPROX_M", F.lit(R) * c)
        if log_this_leg:
            log_shape(cand_hav, f"Leg[{leg_idx}] cand_hav (+D_APPROX_M)", do_explain=False)

        # ---------- EARLY radius filter to shrink data -----------------------
        within = F.col("D_APPROX_M") <= F.lit(radius_m)
        cand_within = cand_hav.filter(within)
        if log_this_leg:
            log_shape(cand_within, f"Leg[{leg_idx}] cand_within (<= radius)", do_explain=False)

        # ---------- Dedupe using aggregation (cheaper than window) -----------
        keys = ["LOAD_NO", "FROM_SEQ", "TO_SEQ", "DOT_NAME"]
        mins = (
            cand_within.group_by(*keys)
            .agg(F.min(F.col("D_APPROX_M")).alias("D_MIN"))
        )
        cw = cand_within.alias("CW")
        m  = mins.alias("M")
        
        nearest = (
            cw.join(
                m,
                (cw["LOAD_NO"]  == m["LOAD_NO"])  &
                (cw["FROM_SEQ"] == m["FROM_SEQ"]) &
                (cw["TO_SEQ"]   == m["TO_SEQ"])   &
                (cw["DOT_NAME"] == m["DOT_NAME"]) &
                (cw["D_APPROX_M"] == m["D_MIN"]),
                how="inner",
            )
            # ---- restore original column names so downstream .select() works ----
            .select(*[cw[c].alias(c) for c in cw.columns])
        )

        if log_this_leg:
            log_shape(nearest, f"Leg[{leg_idx}] nearest (agg-based)", do_explain=EXPLAIN)

        # ---------- Finalize projection --------------------------------------
        final_df = nearest.select(
            "LOAD_NO","FROM_SEQ","TO_SEQ","STATE",
            F.col("DOT_LON").alias("LON"),
            F.col("DOT_LAT").alias("LAT"),
            F.col("MM").alias("MM_APPROX_MI"),
            (F.col("D_APPROX_M") / F.lit(METERS_PER_MILE)).alias("DIST_MI"),
            "DOT_NAME",
        )


        if log_this_leg:
            log_shape(final_df, f"Leg[{leg_idx}] final_df (within radius)", do_explain=EXPLAIN)

        try:
            out_pd = final_df.to_pandas()
        except Exception:
            logger.error(f"[{run_id}] Leg[{leg_idx}] to_pandas failed (see incident in Query History)\n{traceback.format_exc()}")
            # force execution separately to capture qid even if pandas bridge failed
            try:
                _ = final_df.count()
                logger.info(f"[{run_id}] Leg[{leg_idx}] final_df.count() succeeded; qid={_last_qid()}")
            except Exception:
                logger.error(f"[{run_id}] Leg[{leg_idx}] final_df.count() also failed; qid={_last_qid()}")
            raise

        if not out_pd.empty:
            out_pd.sort_values(["LOAD_NO","FROM_SEQ","TO_SEQ","MM_APPROX_MI"], inplace=True)
            out_pd["EVENT_TYPE"] = "DC"
            out_pd["IS_DC"] = True
            out_chunks.append(out_pd)
        elif log_this_leg:
            logger.info(f"[{run_id}] Leg[{leg_idx}] final: 0 rows within {radius_mi} mi")

    if out_chunks:
        result = pd.concat(out_chunks, ignore_index=True)
        logger.info(f"[{run_id}] DONE: concatenated chunks -> {len(result)} rows")
        return result

    logger.info(f"[{run_id}] DONE: no DC candidates produced")
    # empty but schema-consistent when no matches
    return pd.DataFrame(columns=[
        "LOAD_NO","FROM_SEQ","TO_SEQ","STATE","LON","LAT",
        "MM_APPROX_MI","DIST_MI","DOT_NAME","EVENT_TYPE","IS_DC"
    ])


# ---- cell 8 ----
def _coerce_payload_to_dict(obj):
    """Return a Python dict for the payload regardless of how Snowpark returns VARIANT."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        return json.loads(obj)
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        raise TypeError(f"Expected dict-like payload; got {type(obj).__name__}")

def _dequote_once(s: str) -> str:
    if not s: return s
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def _undollar(s: str) -> str:
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        return s[2:-2]
    return s

def _normalize_args(argv):
    out = []
    for raw in argv or []:
        s = str(raw)
        s = _dequote_once(_undollar(s)).strip()
        if s: out.append(s)
    return out

_RUNID_RE = re.compile(r"^\s*RUN_ID\s*=\s*(.+?)\s*$", re.IGNORECASE)

def _extract_run_id(argv):
    for s in _normalize_args(argv):
        m = _RUNID_RE.match(s)
        if m:
            return m.group(1).strip()
    return None

def _extract_run_json(argv):
    args = _normalize_args(argv)
    # RUN_JSON={run_json}
    for s in args:
        if s.upper().startswith("RUN_JSON="):
            return s.split("=", 1)[1].strip()
    # Single positional JSON
    for s in args:
        if s.startswith("{") and s.endswith("}"):
            return s
    return None

def _get_run_payload(argv, session):
    args = _normalize_args(argv)
    rid  = _extract_run_id(args)
    if rid:
        rows = (session.table("DEV_DOT_MLAI.DTI_FUEL.FUEL_RUN_INBOX")
                      .filter(col("RUN_ID") == rid)
                      .select("PAYLOAD")
                      .limit(1)
                      .collect())
        if not rows:
            raise ValueError(f"RUN_ID '{rid}' not found in DEV_DOT_MLAI.DTI_FUEL.FUEL_RUN_INBOX")
        return _coerce_payload_to_dict(rows[0][0])

    j = _extract_run_json(args)
    if j:
        return json.loads(j)

    raise ValueError("No RUN_ID or JSON argument found. argv(normalized)=" + repr(args))


def loads_json_or_none(s):
    if not s or s in ("None", ""): return None
    if isinstance(s, dict): return s
    return json.loads(s)

def get_num(v, typ, default):
    try:
        if v in (None, "", "None"): return default
        return typ(v)
    except Exception:
        return default

def parse_ts(s):
    if not s or s in ("None", ""): return None
    return datetime.fromisoformat(s.replace(" ", "T").replace("Z", ""))

STOP_SCHEMA = StructType([
    StructField("LOAD_NUMBER", IntegerType()),
    StructField("LOAD_STOP_SEQUENCE_NO", IntegerType()),
    StructField("APPT_ETA_TS", TimestampType()),
    StructField("STOP_NAME", StringType()),
    StructField("ADDRESS1", StringType()),
    StructField("CITY", StringType()),
    StructField("STATE", StringType()),
    StructField("POSTALCODE", StringType()),
    StructField("LON", DoubleType()),
    StructField("LAT", DoubleType()),
    StructField("DRIVER_ONE_EMP_NO", IntegerType()),
    StructField("DRIVER_TWO_EMP_NO", IntegerType()),
    StructField("DOT_DOMICILEABBREVIATION", StringType()),
])

@dataclass(frozen=True)
class FuelPlanInputs:
    load_number: int
    initial_fuel: float
    tank_capacity: float
    mpg: float
    safety_buffer: float
    stops_df: Optional[Any] = None
    raw_payload: Any = None       
    run_id: str = ""              

def build_inputs_from_argv(argv: List[str], session: Session) -> FuelPlanInputs:
    run = _get_run_payload(argv, session)
    rid = _extract_run_id(argv) or uuid.uuid4().hex

    load_number   = get_num(run.get("LOAD_NUMBER"), int, None)
    initial_fuel  = get_num(run.get("INITIAL_FUEL"), float, None)
    tank_capacity = get_num(run.get("TANK_CAPACITY"), float, None)
    mpg           = get_num(run.get("MPG"), float, None)
    safety_buffer = get_num(run.get("SAFETY_BUFFER"), float, None)

    missing = [k for k, v in {
        "LOAD_NUMBER": load_number,
        "INITIAL_FUEL": initial_fuel,
        "TANK_CAPACITY": tank_capacity,
        "MPG": mpg,
        "SAFETY_BUFFER": safety_buffer,
    }.items() if v is None]
    if missing:
        raise ValueError(f"RUN_JSON missing required run-level field(s): {', '.join(missing)}")

    stops_in = run.get("STOPS")
    if not isinstance(stops_in, list) or not stops_in:
        raise ValueError("RUN_JSON must contain a non-empty STOPS array.")
    if len(stops_in) < 2:
        raise ValueError("STOPS must contain at least an origin and a destination.")

    norm = []
    for s in stops_in:
        seq = get_num(s.get("LOAD_STOP_SEQUENCE_NO"), int, 0)
        norm.append({
            "LOAD_NUMBER":        load_number,
            "LOAD_STOP_SEQUENCE_NO": seq,
            "APPT_ETA_TS":        parse_ts(s.get("LOAD_STOP_SCHEDULEDSTARTDATE")),
            "STOP_NAME":          (s.get("DOT_DOMICILENAME") or s.get("STOP_NAME") or ""),
            "ADDRESS1":           (s.get("ADDRESSONE") or s.get("ADDRESS1") or ""),
            "CITY":               (s.get("CITY") or ""),
            "STATE":              (s.get("STATE") or ""),
            "POSTALCODE":         (s.get("POSTALCODE") or ""),
            "LON":                get_num(s.get("LON"), float, None),
            "LAT":                get_num(s.get("LAT"), float, None),
            "DRIVER_ONE_EMP_NO":  get_num(s.get("DRIVER_ONE_EMPLOYEE_NUMBER"), int, 0),
            "DRIVER_TWO_EMP_NO":  get_num(s.get("DRIVER_TWO_EMPLOYEE_NUMBER"), int, 0),
            "DOT_DOMICILEABBREVIATION": (s.get("DOT_DOMICILEABBREVIATION") or ""),
        })

    norm.sort(key=lambda r: r["LOAD_STOP_SEQUENCE_NO"])
    stops_df = session.create_dataframe(norm, schema=STOP_SCHEMA)
    
    return FuelPlanInputs(
        load_number=load_number,
        initial_fuel=initial_fuel,
        tank_capacity=tank_capacity,
        mpg=mpg,
        safety_buffer=safety_buffer,
        stops_df=stops_df,
        raw_payload=run,
        run_id=rid
    )


# ---- cell 9 ----
def _first(d, *names):
    for n in names:
        v = d.get(n)
        if v is not None:
            return v
    return None

def get_hos_for_driver(driver_id: str) -> dict:
    r = requests.get(f"{BASE}/fleet/hos/clocks",
                     params={"driverIds": driver_id, "limit": 1},
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return {}

    rec = data[0]
    drive_s = _first(rec, "availableDriveSeconds", "remainingDriveSeconds")
    if drive_s is None:
        ms = _first(rec, "driveRemainingMs", "availableDriveMs", "remainingDriveMs")
        drive_s = ms / 1000.0 if ms is not None else None

    shift_s = _first(rec, "availableShiftSeconds", "remainingShiftSeconds")
    if shift_s is None:
        ms = _first(rec, "shiftRemainingMs", "availableShiftMs", "remainingShiftMs")
        shift_s = ms / 1000.0 if ms is not None else None

    cycle_s = _first(rec, "availableCycleSeconds", "remainingCycleSeconds", "cycleRemainingMs")
    if isinstance(cycle_s, (int, float)) and cycle_s > 10000 and "Ms" in "".join(rec.keys()):
        # crude guard if cycle_s actually came back in ms
        cycle_s = cycle_s / 1000.0

    return {
        "drive_hours": None if drive_s is None else round(drive_s / 3600.0, 2),
        "shift_hours": None if shift_s is None else round(shift_s / 3600.0, 2),
        "cycle_hours": None if cycle_s is None else round(cycle_s / 3600.0, 2),
        "raw": rec,
    }

_EMPNO_RX = re.compile(r"\((\d+)\)\s*$")

def _parse_empno_from_name(name: str) -> str | None:
    if not name:
        return None
    m = _EMPNO_RX.search(str(name))
    return m.group(1) if m else None

def find_driver_id_by_empno(emp_no: int | str) -> str | None:
    """Iterate /fleet/drivers and match '(empno)' at the end of the name."""
    emp_no = str(emp_no).strip()
    after = None
    for _ in range(100):  # hard cap to avoid runaway loops
        params = {"limit": 200}
        if after:
            params["after"] = after
        r = requests.get(f"{BASE}/fleet/drivers", params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or j.get("drivers") or []
        for d in data:
            n = d.get("name") or ""
            parsed = _parse_empno_from_name(n)
            if parsed == emp_no:
                did = d.get("id") or d.get("driverId")
                return str(did) if did is not None else None
        pg = j.get("pagination") or {}
        has_next = bool(pg.get("hasNextPage"))
        after = pg.get("endCursor")
        if not has_next or not after:
            break
    return None

def get_start_drive_hours_for_load(session, stops_sp) -> float:
    """
    Look at the first stop’s drivers (DRIVER_ONE_EMP_NO / DRIVER_TWO_EMP_NO),
    map to Samsara driverId(s), fetch HOS clocks, and return remaining drive hours.
    For team, we use the max of the two as a conservative “you can still drive” number.
    Fallback to 10.0h if anything is missing.
    """
    try:
        first = (
            stops_sp.sort(F.col("LOAD_STOP_SEQUENCE_NO").asc())
                    .select("DRIVER_ONE_EMP_NO","DRIVER_TWO_EMP_NO")
                    .limit(1).to_pandas()
        )
        emp_candidates = []
        if not first.empty:
            d1 = first.iloc[0].get("DRIVER_ONE_EMP_NO")
            d2 = first.iloc[0].get("DRIVER_TWO_EMP_NO")
            for v in (d1, d2):
                if v and int(v) > 0:
                    emp_candidates.append(int(v))

        hours = []
        for emp in emp_candidates:
            did = find_driver_id_by_empno(emp)
            if did:
                hos = get_hos_for_driver(did)
                if hos and hos.get("drive_hours") is not None:
                    hours.append(float(hos["drive_hours"]))
        if hours:
            # For solo this is just the solo’s hours; for team take the max.
            return round(max(hours), 2)
    except Exception as e:
        print(f"[HOS] Falling back to default start_drive_hours; reason: {e}")
    return 10.0

# ---- cell 10 ----
DTI_FUEL_DATA_SUMMARY_SP.show(5)

# ---- cell 11 ----
DC_FUEL_DATA_SP.show(5)

# ---- cell 12 ----
dc   = DC_FUEL_DATA_SP.alias("dc")
summ = DTI_FUEL_DATA_SUMMARY_SP.select("STATE", "STATE_TAX_MODE").alias("summ")

# Join on STATE and compute net-of-tax price
dc_fuel_with_mode = (
    dc.join(summ, dc["STATE"] == summ["STATE"], how="left")
      .with_column(
          "PPG_NO_TAX",
          dc["PRICEPERGALLON"].cast("FLOAT")
          - F.coalesce(summ["STATE_TAX_MODE"].cast("FLOAT"), F.lit(0.0))
      )
      .select(
          *[dc[c] for c in dc.columns],
          summ["STATE_TAX_MODE"],
          F.col("PPG_NO_TAX")
      )
)

# ---- cell 13 ----
dc_fuel_with_mode.show(5)

# ---- cell 14 ----
def authenticate_fleet(api_key: str, account_id=None) -> str:
    url = "https://fleets.trimblemaps.com/api/assets/v1/accounts/authenticate"
    payload = {"apiKey": api_key}
    if account_id:
        payload["accountId"] = account_id
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError(f"Identity returned 200 but no token: {resp.text}")
    return token

# ---- cell 15 ----
def fetch_avoid_favor_sets(api_key: str, use_vehicle_group: bool = False) -> list[str]:
    jwt = authenticate_fleet(api_key)
    headers = {"Authorization": f"Bearer {jwt}"}
    url = f"{AVOIDFAVOR_BASE_URL}?limit=1000&includeDefaultClosure=true"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    all_sets = resp.json().get("data", [])
    prod_sets = [
        str(s["setId"])
        for s in all_sets
        if any(g["id"] == int(DTI_VEHICLE_GROUP_ID) for g in s.get("vehicleGroups", []))
    ]
    if DTI_DEFAULT_CLOSURE_ID not in prod_sets:
        prod_sets.append(DTI_DEFAULT_CLOSURE_ID)
    return prod_sets

# ---- cell 16 ----
def get_route_mileage(f_lon, f_lat, t_lon, t_lat, api_key, af_set_ids):
    params = {
        "stops":     f"{f_lon},{f_lat};{t_lon},{t_lat}",
        "afSetIDs":  ",".join(af_set_ids),
        "reports":   "CalcMiles",
        "profileId": DEFAULT_PROFILE_ID,
        "authToken": api_key
    }
    url = f"{ROUTE_REPORTS_BASE_URL}?{urllib.parse.urlencode(params)}"
    last_exc = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            j = r.json()
            return float(j[0]["TMiles"])
        except Exception as e:
            print(f"[CalcMiles] attempt {attempt} failed url={url} err={e} body={getattr(r, 'text', '')[:500]}")
            last_exc = e
            time.sleep(1)
    raise RuntimeError(f"PC*MILER CalcMiles failed after retries: {last_exc}")

# ---- cell 17 ----
def create_mileage_cache(api_key: str):
    # Pull AF set IDs once per kernel; memoize route calls
    af_ids = tuple(fetch_avoid_favor_sets(api_key))
    @lru_cache(maxsize=8192)
    def _pcm(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
        return get_route_mileage(lon_a, lat_a, lon_b, lat_b, api_key, af_ids)
    return _pcm

# ---- cell 18 ----
def build_legs_and_compute_mileage(session: Session, stops_df, load_number: int):
    """
    Input: Snowpark DF (stops_df) with columns at least:
      LOAD_NUMBER (int), LOAD_STOP_SEQUENCE_NO (int), LON (float), LAT (float)

    Output:
      - Snowpark DF with legs and mileage
      - If persist=True, writes TMP_LEG_MILES_<load_number> and returns (df, table_name)
    """
    from snowflake.snowpark.types import StructType, StructField, IntegerType, DoubleType
    
    cols = {c.upper() for c in stops_df.columns}
    required = {"LOAD_NUMBER","LOAD_STOP_SEQUENCE_NO","LON","LAT"}
    missing = required - cols
    if missing:
        raise ValueError(f"stops_df missing required columns: {sorted(missing)}")

    pdf = (stops_df
           .select("LOAD_NUMBER","LOAD_STOP_SEQUENCE_NO","LON","LAT")
           .to_pandas()
           .sort_values(["LOAD_NUMBER","LOAD_STOP_SEQUENCE_NO"])
           .reset_index(drop=True))

    # Prime mileage function
    if not PC_MILER_API_KEY:
        raise RuntimeError("PC_MILER_API_KEY not found (env or Snowflake secret).")
    mileage_fn = create_mileage_cache(PC_MILER_API_KEY)

    # Build legs and compute miles
    legs = []
    for load_no, grp in pdf.groupby("LOAD_NUMBER", sort=False):
        grp = grp.sort_values("LOAD_STOP_SEQUENCE_NO").reset_index(drop=True)
        for i in range(len(grp) - 1):
            a = grp.iloc[i]; b = grp.iloc[i+1]
            lon_a, lat_a, lon_b, lat_b = a["LON"], a["LAT"], b["LON"], b["LAT"]
            miles = None
            if pd.notna(lon_a) and pd.notna(lat_a) and pd.notna(lon_b) and pd.notna(lat_b):
                try:
                    miles = mileage_fn(float(lon_a), float(lat_a), float(lon_b), float(lat_b))
                except Exception as e:
                    print(f"[LegError] {load_no} {a['LOAD_STOP_SEQUENCE_NO']}→{b['LOAD_STOP_SEQUENCE_NO']} :: {e}")
                    miles = None
            legs.append({
                "LOAD_NUMBER": int(load_no),
                "FROM_SEQ":    int(a["LOAD_STOP_SEQUENCE_NO"]),
                "TO_SEQ":      int(b["LOAD_STOP_SEQUENCE_NO"]),
                "FROM_LON":    float(lon_a) if pd.notna(lon_a) else None,
                "FROM_LAT":    float(lat_a) if pd.notna(lat_a) else None,
                "TO_LON":      float(lon_b) if pd.notna(lon_b) else None,
                "TO_LAT":      float(lat_b) if pd.notna(lat_b) else None,
                "MILES_PCMILER": miles
            })

    schema = StructType([
        StructField("LOAD_NUMBER", IntegerType()),
        StructField("FROM_SEQ", IntegerType()),
        StructField("TO_SEQ", IntegerType()),
        StructField("FROM_LON", DoubleType()),
        StructField("FROM_LAT", DoubleType()),
        StructField("TO_LON", DoubleType()),
        StructField("TO_LAT", DoubleType()),
        StructField("MILES_PCMILER", DoubleType()),
    ])
    
    rows = [
        (
            r["LOAD_NUMBER"], r["FROM_SEQ"], r["TO_SEQ"],
            r["FROM_LON"], r["FROM_LAT"], r["TO_LON"], r["TO_LAT"],
            r["MILES_PCMILER"]
        )
        for r in legs
    ]
    
    legs_sp = session.create_dataframe(rows, schema=schema)
    return legs_sp

# ---- cell 19 ----
def _load_dot_locations_dict(session) -> dict[str, str]:
    rows = (session.table("DEV_DOT_MLAI.DTI_FUEL.DTI_DOT_LOCATIONS")
                 .select("DOT_NAME","LAT","LON")
                 .collect())
    # dict: {"Name": "lat, lon"}
    return {r["DOT_NAME"]: f"{float(r['LAT'])}, {float(r['LON'])}" for r in rows}

# ---- cell 20 ----
def _classify_points_to_states(session, points: list[tuple[float, float]]):
    if not points:
        return []
    values_sql = " , ".join([f"({i},{lon},{lat})" for i,(lon,lat) in enumerate(points)])
    q = f"""
    WITH pts(seq, lon, lat) AS (
      SELECT * FROM VALUES {values_sql}
    )
    SELECT p.seq, p.lon, p.lat, s.name AS state_name
    FROM pts p
    LEFT JOIN DEV_DOT_MLAI.DTI_FUEL.STATES_RAW s
      ON ST_CONTAINS(s.GEOG, TO_GEOGRAPHY(ST_POINT(p.lon, p.lat)))
    ORDER BY p.seq
    """
    df = session.sql(q).to_pandas()
    return list(df["STATE_NAME"].fillna("UNKNOWN"))

# ---- cell 21 ----
def get_route_geotunnel(f_lon, f_lat, t_lon, t_lat, api_key):
    """
    Return a list of (lon, lat) tuples along the routed path between A->B.
    """
    try:
        af_ids = fetch_avoid_favor_sets(api_key)
    except Exception:
        af_ids = []

    params = {
        "stops":     f"{f_lon},{f_lat};{t_lon},{t_lat}",
        "reports":   "RoutePath",
        "profileId": DEFAULT_PROFILE_ID,
        "authToken": api_key
    }
    if af_ids:
        # BUGFIX: add the dot before join + coerce to str
        params["afSetIDs"] = ",".join(map(str, af_ids))

    url = f"{ROUTE_REPORTS_BASE_URL}?{urllib.parse.urlencode(params)}"

    last_exc = None
    for attempt in range(1, 4):
        t0 = time.time()
        try:
            _debug(f"RoutePath attempt={attempt} url={url}")
            r = requests.get(url, timeout=30)
            _debug(f"status={r.status_code} ct={r.headers.get('content-type')} t_ms={(time.time()-t0)*1000:.1f} len={len(r.text)}")
            r.raise_for_status()
            j = r.json()

                        # Normalize top node
            top = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
            keys = list(top.keys()) if isinstance(top, dict) else []
            _debug(f"top_type={type(top).__name__} keys={keys[:20]}")

            #handle GeoJSON-style payloads
            pts_raw = None
            if isinstance(top, dict):
                geom = top.get("geometry")
                if isinstance(geom, dict):
                    gtype = (geom.get("type") or "").lower()
                    coords = geom.get("coordinates")
                    if coords:
                        # GeoJSON is [lon, lat]
                        if gtype == "linestring":
                            pts_raw = coords  # [[lon, lat], ...]
                            _debug(f"geometry=LineString coords_len={len(coords)}")
                        elif gtype == "multilinestring":
                            # Flatten list of lines
                            pts_raw = [pt for line in coords for pt in line]
                            _debug(f"geometry=MultiLineString segments={len(coords)} total_pts={len(pts_raw)}")

            if pts_raw is None and isinstance(top, dict):
                for k in ("Path", "Shape", "Polyline", "Coords"):
                    v = top.get(k)
                    if v:
                        _debug(f"legacy candidate {k} length={len(v)} first2={str(v[:2])[:200]}")
                        pts_raw = [{"Lon": p.get("Lon") or p.get("Longitude"),
                                    "Lat": p.get("Lat") or p.get("Latitude")} for p in v]
                        break

            if pts_raw is None:
                raise RuntimeError(f"GeoTunnel empty; top_keys={keys[:20]}")

            # Canonicalize to list[(lon, lat)]
            out = []
            # If GeoJSON (list[list[2 floats]]), normalize
            if pts_raw and isinstance(pts_raw[0], (list, tuple)):
                for lon, lat in pts_raw:
                    if lon is not None and lat is not None:
                        out.append((float(lon), float(lat)))
            else:
                # Legacy dict form with Lon/Lat keys
                for p in pts_raw:
                    lon = p.get("Lon")
                    lat = p.get("Lat")
                    if lon is not None and lat is not None:
                        out.append((float(lon), float(lat)))

            _debug(f"parsed_points={len(out)} f=({f_lon},{f_lat}) t=({t_lon},{t_lat})")
            if not out:
                raise RuntimeError(f"GeoTunnel parsed 0 points; top_keys={keys[:20]}")
            return out


        except Exception as e:
            preview = ""
            try:
                preview = r.text[:500]
            except Exception:
                pass
            _debug(f"[RoutePath] attempt={attempt} failed err={e} body={preview}")
            last_exc = e
            time.sleep(1)

    raise RuntimeError(f"PC*MILER RoutePath failed after retries: {last_exc}")

# ---- cell 22 ----
def _norm_state(s: str) -> str:
    if not s: return ""
    return STATE_TO_ABBR.get(s, s).upper()

def _make_transitions(points: list[tuple[float, float]], states: list[str]):
    from math import radians, sin, cos, asin, sqrt
    def hav_mi(lat1, lon1, lat2, lon2):
        R = 3958.7613
        dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 2*R*asin(sqrt(a))

    out = []
    if not points or not states:
        return out

    # cumulative miles along the tunnel
    mm = 0.0
    prev_state = _norm_state(states[0])
    prev_lon, prev_lat = points[0]
    for i in range(1, len(points)):
        lon, lat = points[i]
        mm += hav_mi(prev_lat, prev_lon, lat, lon)
        cur_state = _norm_state(states[i])
        if cur_state and prev_state and cur_state != prev_state:
            out.append({
                "from_state": prev_state,
                "to_state": cur_state,
                "transition_lon": float(lon),
                "transition_lat": float(lat),
                "mm": float(mm), # mile marker
            })
        prev_state = cur_state
        prev_lon, prev_lat = lon, lat
    return out



# ---- cell 23 ----
def build_routes_df_for_events(session, stops_sp, legs_sp, pcmiler_api_key: str) -> "pd.DataFrame":
    # Normalize stops (unchanged)
    stops_pd = (
        stops_sp.select(
            F.col("LOAD_NUMBER").alias("LOAD_NO"),
            F.col("LOAD_STOP_SEQUENCE_NO").alias("LOAD_STOP_SEQUENCE_NO"),
            "CITY","STATE",
            F.col("LON").alias("LON"),
            F.col("LAT").alias("LAT"),
            F.col("DOT_DOMICILEABBREVIATION").alias("DOT_DOMICILE_ABBRV")
        ).to_pandas()
    )
    stops_pd.columns = [str(c).strip().upper() for c in stops_pd.columns]
    print("stops_pd columns:", list(stops_pd.columns))
    stops_pd = stops_pd.sort_values(["LOAD_NO","LOAD_STOP_SEQUENCE_NO"]).reset_index(drop=True)

    # Init per-leg fields
    stops_pd["GEOTUNNEL_POINTS"]  = None
    stops_pd["STATE_TRANSITIONS"] = None
    # carry per-leg scaling info for downstream DC sampling
    stops_pd["LEG_MI_PCM"] = None
    stops_pd["LEG_SCALE"]  = None

    # Bring leg miles
    legs_pd = (
        legs_sp.select("LOAD_NUMBER","FROM_SEQ","TO_SEQ","FROM_LON","FROM_LAT","TO_LON","TO_LAT","MILES_PCMILER")
               .to_pandas()
               .sort_values(["LOAD_NUMBER","FROM_SEQ"])
               .reset_index(drop=True)
    )
    leg_mi_map = {
        (int(r.LOAD_NUMBER), int(r.FROM_SEQ)): (float(r.MILES_PCMILER) if pd.notna(r.MILES_PCMILER) else None)
        for r in legs_pd.itertuples(index=False)
    }

    for r in legs_pd.itertuples(index=False):
        load_no  = int(r.LOAD_NUMBER)
        from_seq = int(r.FROM_SEQ)
        tunnel = get_route_geotunnel(
            f_lon=float(r.FROM_LON), f_lat=float(r.FROM_LAT),
            t_lon=float(r.TO_LON),   t_lat=float(r.TO_LAT),
            api_key=pcmiler_api_key
        )
        # Classify states + build transitions with raw mm
        states = _classify_points_to_states(session, tunnel)
        transitions = _make_transitions(tunnel, states)  # has 'mm' relative to leg start

        # scale leg mm to PC*Miler miles
        leg_pcm_mi = leg_mi_map.get((load_no, from_seq))
        gt_len_mi  = _polyline_length_mi(tunnel)
        scale = (float(leg_pcm_mi) / gt_len_mi) if (leg_pcm_mi and gt_len_mi) else 1.0
        for t in transitions:
            if "mm" in t and t["mm"] is not None:
                t["mm"] = float(t["mm"]) * float(scale)

        if DEBUG_PCM:
            print(f"[PCM-DEBUG] load={load_no} leg={from_seq}->{int(r.TO_SEQ)} "
                  f"pts={len(tunnel)} trans={len(transitions)} scale={scale:.6f}")

        match = (stops_pd["LOAD_NO"] == load_no) & (stops_pd["LOAD_STOP_SEQUENCE_NO"] == from_seq)
        if match.any():
            idx = stops_pd.index[match][0]
            stops_pd.at[idx, "GEOTUNNEL_POINTS"]  = tunnel
            stops_pd.at[idx, "STATE_TRANSITIONS"] = transitions
            stops_pd.at[idx, "LEG_MI_PCM"]        = leg_pcm_mi
            stops_pd.at[idx, "LEG_SCALE"]         = scale

    return stops_pd


# ---- cell 24 ----
def process_routes(
    session: Session,
    legs_sp, # DF: LOAD_NUMBER, FROM_SEQ, TO_SEQ, FROM_LON, FROM_LAT, TO_LON, TO_LAT, MILES_PCMILER
    *,
    sample_km: float = 10.0,
    max_samples_per_leg: int = 500,
    states_table: str = "DEV_DOT_MLAI.DTI_FUEL.STATES_RAW",
    dot_table: str = "DEV_DOT_MLAI.DTI_FUEL.DTI_DOT_LOCATIONS",
):
    import uuid
    # unwrap (df, table_name)
    if isinstance(legs_sp, tuple):
        legs_sp = legs_sp[0]

    tmp_legs = f"TEMP_LEGS_{uuid.uuid4().hex[:12]}"
    legs_sp.write.mode("overwrite").save_as_table(tmp_legs, table_type="temporary")

    q = f"""
    WITH
    legs AS (
      SELECT
        LOAD_NUMBER, FROM_SEQ, TO_SEQ,
        FROM_LON, FROM_LAT, TO_LON, TO_LAT, MILES_PCMILER,
        ST_MAKELINE(
          TO_GEOGRAPHY(ST_POINT(FROM_LON, FROM_LAT)),
          TO_GEOGRAPHY(ST_POINT(TO_LON, TO_LAT))
        ) AS GEOG_LINE
      FROM {tmp_legs}
    ),
    legs_sized AS (
      SELECT
        *,
        ST_LENGTH(GEOG_LINE) AS LINE_M,
        LEAST(
          {max_samples_per_leg},
          GREATEST(2, FLOOR(ST_LENGTH(GEOG_LINE) / ({sample_km}*1000)) + 1)
        ) AS N_SAMPLES
      FROM legs
    ),
    -- GENERATOR must be constant; generate 0..max-1 and keep < N_SAMPLES per leg
    gen AS (SELECT SEQ4() AS G FROM TABLE(GENERATOR(ROWCOUNT => {max_samples_per_leg}))),
    samples AS (
      SELECT
        l.LOAD_NUMBER, l.FROM_SEQ, l.TO_SEQ,
        l.FROM_LON, l.FROM_LAT, l.TO_LON, l.TO_LAT,
        l.MILES_PCMILER, l.GEOG_LINE, l.LINE_M, l.N_SAMPLES,
        g.G AS IDX
      FROM legs_sized l
      JOIN gen g ON g.G < l.N_SAMPLES
    ),
    pts AS (
      SELECT
        LOAD_NUMBER, FROM_SEQ, TO_SEQ, MILES_PCMILER, GEOG_LINE, LINE_M, N_SAMPLES, IDX,
        (IDX::DOUBLE) / NULLIF(N_SAMPLES - 1, 0) AS FRACTION,
        -- replace ST_GEODESICINTERPOLATE/LINE_INTERPOLATE with simple lon/lat lerp
        TO_GEOGRAPHY(
          ST_POINT(
            FROM_LON + ((TO_LON - FROM_LON) * ((IDX::DOUBLE) / NULLIF(N_SAMPLES - 1, 0))),
            FROM_LAT + ((TO_LAT - FROM_LAT) * ((IDX::DOUBLE) / NULLIF(N_SAMPLES - 1, 0)))
          )
        ) AS PT,
        ((IDX::DOUBLE) / NULLIF(N_SAMPLES - 1, 0)) * LINE_M AS PT_M
      FROM samples
    ),
    pts_state AS (
      SELECT p.*, s.NAME AS STATE_NAME
      FROM pts p
      LEFT JOIN {states_table} s
        ON ST_CONTAINS(s.GEOG, p.PT)
    ),
    -- two-step window to avoid nested window error
    runs1 AS (
      SELECT
        *,
        COALESCE(STATE_NAME, 'UNKNOWN') AS STATE_NM_NN,
        LAG(COALESCE(STATE_NAME,'UNKNOWN')) OVER (
          PARTITION BY LOAD_NUMBER, FROM_SEQ, TO_SEQ
          ORDER BY IDX
        ) AS STATE_PREV
      FROM pts_state
    ),
    runs2 AS (
      SELECT
        *,
        CASE WHEN STATE_NM_NN <> STATE_PREV THEN 1 ELSE 0 END AS CHG
      FROM runs1
    ),
    runs AS (
      SELECT
        *,
        SUM(CHG) OVER (
          PARTITION BY LOAD_NUMBER, FROM_SEQ, TO_SEQ
          ORDER BY IDX
          ROWS UNBOUNDED PRECEDING
        ) AS RUN_ID
      FROM runs2
    ),
    segs AS (
      SELECT
        LOAD_NUMBER, FROM_SEQ, TO_SEQ,
        STATE_NM_NN AS STATE,
        MIN(PT_M)  AS START_M,
        MAX(PT_M)  AS END_M,
        MIN(IDX)   AS START_IDX,
        MAX(IDX)   AS END_IDX,
        MAX(LINE_M) AS LINE_M
      FROM runs
      GROUP BY LOAD_NUMBER, FROM_SEQ, TO_SEQ, STATE_NM_NN, RUN_ID
    ),
    segs_miles AS (
      SELECT s.*, (END_M - START_M) / {METERS_PER_MILE} AS SEG_MI
      FROM segs s
    ),
    segs_scaled AS (
      SELECT
        sm.*,
        l.MILES_PCMILER,
        CASE
          WHEN l.MILES_PCMILER IS NOT NULL AND l.MILES_PCMILER > 0 THEN
            ROUND(
              sm.SEG_MI * l.MILES_PCMILER
              / NULLIF(SUM(sm.SEG_MI) OVER (PARTITION BY sm.LOAD_NUMBER, sm.FROM_SEQ, sm.TO_SEQ), 0),
              2
            )
          ELSE ROUND(sm.SEG_MI, 2)
        END AS SEG_MI_SCALED
      FROM segs_miles sm
      JOIN {tmp_legs} l
        ON sm.LOAD_NUMBER = l.LOAD_NUMBER
       AND sm.FROM_SEQ    = l.FROM_SEQ
       AND sm.TO_SEQ      = l.TO_SEQ
    ),
    ends AS (
      SELECT
        l.LOAD_NUMBER, l.FROM_SEQ, l.TO_SEQ,
        TO_GEOGRAPHY(ST_POINT(l.FROM_LON, l.FROM_LAT)) AS FROM_GEOG,
        TO_GEOGRAPHY(ST_POINT(l.TO_LON,   l.TO_LAT))   AS TO_GEOG
      FROM {tmp_legs} l
    ),
    nearest_from AS (
      SELECT
        e.LOAD_NUMBER, e.FROM_SEQ, e.TO_SEQ,
        d.DOT_NAME,
        ROW_NUMBER() OVER (
          PARTITION BY e.LOAD_NUMBER, e.FROM_SEQ, e.TO_SEQ
          ORDER BY ST_DISTANCE(e.FROM_GEOG, d.GEOG)
        ) AS RN
      FROM ends e
      JOIN {dot_table} d
        ON ST_DISTANCE(e.FROM_GEOG, d.GEOG) <= 50000
    ),
    nearest_to AS (
      SELECT
        e.LOAD_NUMBER, e.FROM_SEQ, e.TO_SEQ,
        d.DOT_NAME,
        ROW_NUMBER() OVER (
          PARTITION BY e.LOAD_NUMBER, e.FROM_SEQ, e.TO_SEQ
          ORDER BY ST_DISTANCE(e.TO_GEOG, d.GEOG)
        ) AS RN
      FROM ends e
      JOIN {dot_table} d
        ON ST_DISTANCE(e.TO_GEOG, d.GEOG) <= 50000
    ),
    labeled AS (
      SELECT
        ss.LOAD_NUMBER, ss.FROM_SEQ, ss.TO_SEQ,
        ss.STATE,
        ss.START_IDX, ss.END_IDX, ss.START_M, ss.END_M,
        ss.SEG_MI_SCALED, ss.MILES_PCMILER,
        nf.DOT_NAME AS FROM_DOT,
        nt.DOT_NAME AS TO_DOT
      FROM segs_scaled ss
      LEFT JOIN (SELECT * FROM nearest_from WHERE RN = 1) nf
        ON ss.LOAD_NUMBER = nf.LOAD_NUMBER AND ss.FROM_SEQ = nf.FROM_SEQ AND ss.TO_SEQ = nf.TO_SEQ
      LEFT JOIN (SELECT * FROM nearest_to   WHERE RN = 1) nt
        ON ss.LOAD_NUMBER = nt.LOAD_NUMBER AND ss.FROM_SEQ = nt.FROM_SEQ AND ss.TO_SEQ = nt.TO_SEQ
      ORDER BY LOAD_NUMBER, FROM_SEQ, TO_SEQ, START_IDX
    )
    SELECT * FROM labeled
    """
    return session.sql(q)

# ---- cell 25 ----
def expand_route_events(
    session,
    legs_sp, # DF: LOAD_NUMBER, FROM_SEQ, TO_SEQ, FROM_LON, FROM_LAT, TO_LON, TO_LAT
    *,
    threshold_miles: float = 15.0,
    max_samples_per_leg: int = 500,
    dot_table: str = "DEV_DOT_MLAI.DTI_FUEL.DTI_DOT_LOCATIONS",
):
    """
    Returns Snowpark DF with columns expected by add_event_miles:
      LOAD_NO, EVENT_SEQ, LON, LAT, DIST_MI, FROM_SEQ, TO_SEQ, DOT_NAME
    """
    import uuid
    tmp_legs = f"TEMP_LEGS_{uuid.uuid4().hex[:12]}"
    legs_sp.write.mode("overwrite").save_as_table(tmp_legs, table_type="temporary")

    max_m = threshold_miles * METERS_PER_MILE

    q = f"""
    WITH
    legs AS (
      SELECT
        LOAD_NUMBER, FROM_SEQ, TO_SEQ,
        FROM_LON, FROM_LAT, TO_LON, TO_LAT
      FROM {tmp_legs}
    ),
    legs_sized AS (
      SELECT
        *,
        -- estimate polyline length (straight line) in meters
        ST_LENGTH(
          ST_MAKELINE(
            TO_GEOGRAPHY(ST_POINT(FROM_LON, FROM_LAT)),
            TO_GEOGRAPHY(ST_POINT(TO_LON, TO_LAT))
          )
        ) AS LINE_M,
        -- choose sample count (capped) so we can “walk” the line
        LEAST(
          {max_samples_per_leg},
          GREATEST(2, FLOOR(
            ST_LENGTH(ST_MAKELINE(TO_GEOGRAPHY(ST_POINT(FROM_LON, FROM_LAT)),
                                  TO_GEOGRAPHY(ST_POINT(TO_LON, TO_LAT)))) / (10*1000)
          ) + 1)
        ) AS N_SAMPLES
      FROM legs
    ),
    gen AS (SELECT SEQ4() AS G FROM TABLE(GENERATOR(ROWCOUNT => {max_samples_per_leg}))),
    samples AS (
      SELECT
        l.LOAD_NUMBER, l.FROM_SEQ, l.TO_SEQ,
        l.FROM_LON, l.FROM_LAT, l.TO_LON, l.TO_LAT,
        l.LINE_M, l.N_SAMPLES,
        g.G AS IDX,
        (g.G::DOUBLE) / NULLIF(l.N_SAMPLES - 1, 0) AS FRAC
      FROM legs_sized l
      JOIN gen g ON g.G < l.N_SAMPLES
    ),
    pts AS (
      SELECT
        LOAD_NUMBER, FROM_SEQ, TO_SEQ,
        -- interpolate lon/lat between endpoints (straight-line lerp)
        TO_GEOGRAPHY(
          ST_POINT(
            FROM_LON + ((TO_LON - FROM_LON) * FRAC),
            FROM_LAT + ((TO_LAT - FROM_LAT) * FRAC)
          )
        ) AS PT,
        FRAC * LINE_M AS PT_M
      FROM samples
    ),
    hits AS (
      SELECT
        p.LOAD_NUMBER, p.FROM_SEQ, p.TO_SEQ,
        d.DOT_NAME, d.LAT, d.LON,
        ST_DISTANCE(p.PT, d.GEOG) AS D_M,
        p.PT_M
      FROM pts p
      JOIN {dot_table} d
        ON ST_DISTANCE(p.PT, d.GEOG) <= {max_m}
    ),
    nearest_per_dot AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY LOAD_NUMBER, FROM_SEQ, TO_SEQ, DOT_NAME
          ORDER BY D_M ASC
        ) AS RN
      FROM hits
    ),
    events AS (
      SELECT
        LOAD_NUMBER AS LOAD_NO,
        FROM_SEQ, TO_SEQ, DOT_NAME,
        LAT, LON,
        ROUND(D_M / {METERS_PER_MILE}, 2) AS DIST_MI,
        PT_M
      FROM nearest_per_dot
      WHERE RN = 1
    ),
    numbered AS (
      SELECT
        LOAD_NO, FROM_SEQ, TO_SEQ, DOT_NAME, LAT, LON, DIST_MI,
        ROW_NUMBER() OVER (
          PARTITION BY LOAD_NO, FROM_SEQ, TO_SEQ
          ORDER BY PT_M
        ) AS EVENT_SEQ
      FROM events
    )
    SELECT * FROM numbered
    ORDER BY LOAD_NO, FROM_SEQ, TO_SEQ, EVENT_SEQ
    """
    return session.sql(q)


# ---- cell 26 ----
def add_event_miles(events_df, mileage_fn):
    """
    For each LOAD_NO, compute miles from each event to the next event using `mileage_fn`
    (your PC*MILER-backed function). Leaves the last event in each load with None.

    Requires columns: LOAD_NO, EVENT_SEQ, LON, LAT
    """
    import math
    import pandas as pd

    if events_df.empty:
        return events_df

    # Work on a sorted copy
    df = events_df.sort_values(["LOAD_NO", "EVENT_SEQ"]).copy()
    df["EVENT_MILES_TO_NEXT"] = None

    for _, grp in df.groupby("LOAD_NO", sort=False):
        idxs = grp.index.to_list()
        for cur_idx, next_idx in zip(idxs[:-1], idxs[1:]):
            lon_a = df.at[cur_idx, "LON"]
            lat_a = df.at[cur_idx, "LAT"]
            lon_b = df.at[next_idx, "LON"]
            lat_b = df.at[next_idx, "LAT"]

            miles = None
            # Only call the API when we have good coords
            if (pd.notna(lon_a) and pd.notna(lat_a) and
                pd.notna(lon_b) and pd.notna(lat_b)):
                try:
                    # mileage_fn is your cached PCMILER function
                    miles = float(mileage_fn(float(lon_a), float(lat_a),
                                             float(lon_b), float(lat_b)))
                except Exception:
                    miles = None

            df.at[cur_idx, "EVENT_MILES_TO_NEXT"] = miles

    return df


# ---- cell 27 ----
def reconcile_event_miles(events_df: "pd.DataFrame", tol_mi: float = 0.5) -> "pd.DataFrame":
    """
    Ensure Σ EVENT_MILES_TO_NEXT aligns to last(MM_APPROX_MI) - first(MM_APPROX_MI), per LOAD_NO.
    If the residual exceeds tol_mi, distribute proportionally across positive hops.
    """
    if events_df is None or events_df.empty:
        return events_df
    df = events_df.copy()
    df["EVENT_MILES_TO_NEXT"] = pd.to_numeric(df["EVENT_MILES_TO_NEXT"], errors="coerce").fillna(0.0)

    for load_no, g in df.groupby("LOAD_NO", sort=False):
        g = g.sort_values("EVENT_SEQ")
        mm_total = (pd.to_numeric(g["MM_APPROX_MI"], errors="coerce").bfill().fillna(0.0).iloc[-1]
                    - pd.to_numeric(g["MM_APPROX_MI"], errors="coerce").bfill().fillna(0.0).iloc[0])
        mi_sum   = float(g["EVENT_MILES_TO_NEXT"].sum())
        resid    = float(mm_total - mi_sum)
        if abs(resid) <= float(tol_mi) or mi_sum <= 0.0:
            continue
        # proportional adjustment over positive hops
        pos = g["EVENT_MILES_TO_NEXT"] > 0
        weight = g.loc[pos, "EVENT_MILES_TO_NEXT"].values
        wsum   = weight.sum()
        if wsum <= 0:
            continue
        adj = (weight / wsum) * resid
        df.loc[g.loc[pos].index, "EVENT_MILES_TO_NEXT"] = (g.loc[pos, "EVENT_MILES_TO_NEXT"].values + adj).clip(min=0.0)
    # small cleanup
    df["EVENT_MILES_TO_NEXT"] = df["EVENT_MILES_TO_NEXT"].round(3)
    return df


# ---- cell 28 ----
def build_price_inputs(session):
    summ_sp = session.table("DEV_DOT_MLAI.DTI_FUEL.WEX_FUEL_DATA_SUMMARY")

    # Key WEX medians by normalized 2-letter code (works for 'IL' and 'Illinois')
    summ_pd = (
        summ_sp.select(
            F.col("STATE").alias("STATE"),
            F.col("NET_EX_TAX_MEDIAN").cast("FLOAT").alias("NET_EX_TAX_MEDIAN"),
            F.col("STATE_TAX_MODE").cast("FLOAT").alias("STATE_TAX_MODE"),
        ).to_pandas()
    )
    state_price = {_state_key(str(r["STATE"])): float(r["NET_EX_TAX_MEDIAN"])
                   for _, r in summ_pd.iterrows() if r["NET_EX_TAX_MEDIAN"] is not None}

    # DC prices, joined to WEX tax mode, with corrected Snowpark join signature
    dc_sp   = session.table("DEV_DOT_MLAI.DTI_FUEL.DTI_HOMEFUEL_CURRENT_PRICE")
    dc_cols = {c.upper(): c for c in dc_sp.columns}
    state_col = dc_cols.get("STATEDC", dc_cols.get("STATE"))
    if not state_col:
        raise ValueError("DTI_HOMEFUEL_CURRENT_PRICE missing STATE/STATEDC.")

    dc   = dc_sp.alias("dc")
    summ = (
        summ_sp.select(F.col("STATE").alias("S_STATE"),
                       F.col("STATE_TAX_MODE").cast("FLOAT").alias("STATE_TAX_MODE"))
               .alias("summ")
    )
    
    dc_joined = (
        dc.join(
            summ,
            on=F.upper(dc[state_col]) == F.upper(summ["S_STATE"]),
            how="left",
        )
        .select(
            dc["STOPNUMBER"].alias("STOPNUMBER"),
            dc["CITY"].alias("CITY"),
            dc[state_col].alias("STATE"),
            dc["LATITUDE"].cast("FLOAT").alias("LAT"),
            dc["LONGITUDE"].cast("FLOAT").alias("LON"),
            dc["PRICEPERGALLON"].cast("FLOAT").alias("PPG_RAW"),
            (dc["PRICEPERGALLON"].cast("FLOAT") - F.coalesce(summ["STATE_TAX_MODE"], F.lit(0.0))).alias("PPG_NO_TAX"),
        )
    )

    dc_fuel_df = dc_joined.to_pandas()
    return dc_fuel_df, state_price


# ---- cell 29 ----
def _haversine_miles(lat1, lon1, lat2, lon2):
    # great-circle distance in miles
    R = 3958.7613
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(a))

# ---- cell 30 ----
def append_price_info(events_df: pd.DataFrame, dc_fuel_df: pd.DataFrame, state_price: dict, nearest_within_miles: float = 5.0) -> pd.DataFrame:
    """
    For each event row (expects columns: STATE, LAT, LONG),
    - find nearest DC station in the same STATE and attach its net-of-tax price (PPG_NO_TAX)
    - also attach a WEX state fallback price (NET_EX_TAX_MEDIAN)
    - choose EVENT_PPG as nearest DC price if within `nearest_within_miles`, else fallback to WEX
    """
    if events_df.empty:
        # still add the columns so downstream doesn’t break
        out = events_df.copy()
        for c in ["NEAR_DC_STOPNUMBER","NEAR_DC_CITY","NEAR_DC_STATE","NEAR_DC_PPG_NO_TAX","NEAR_DC_DISTANCE_MI","WEX_STATE_NET_EX_MEDIAN","EVENT_PPG"]:
            out[c] = None
        return out

    # Normalize expected columns
    need = {"STATE","LAT","LON"}
    missing = [c for c in need if c not in events_df.columns]
    if missing:
        raise KeyError(f"append_price_info: events_df missing {missing}. Has {list(events_df.columns)}")

    out = events_df.copy()
    out["NEAR_DC_STOPNUMBER"]      = None
    out["NEAR_DC_CITY"]            = None
    out["NEAR_DC_STATE"]           = None
    out["NEAR_DC_PPG_NO_TAX"]      = None
    out["NEAR_DC_DISTANCE_MI"]     = None
    out["WEX_STATE_NET_EX_MEDIAN"] = out["STATE"].map(state_price).astype(float)

    # Normalize event states
    out["STATE_KEY"] = out["STATE"].map(_state_key)

    # Normalize DC price states
    dc_fuel_df = dc_fuel_df.copy()
    dc_fuel_df["STATE_KEY"] = dc_fuel_df["STATE"].apply(_state_key)

    # Index DCs by normalized key
    dc_by_state = {st: grp.reset_index(drop=True)
                   for st, grp in dc_fuel_df.groupby("STATE_KEY", dropna=False)}

    # Use STATE_KEY for nearest-DC lookup
    for i, r in out.iterrows():
        st = r["STATE_KEY"]
        lat = r["LAT"]; lon = r["LON"]
        best_dist = None; best_idx = None; best_df = None

        if st and st in dc_by_state and pd.notna(lat) and pd.notna(lon):
            cand = dc_by_state[st]
            # compute distances
            dists = cand.apply(lambda rr: _haversine_miles(lat, lon, rr["LAT"], rr["LON"]), axis=1)
            j = int(dists.idxmin()) if len(dists) else None
            if j is not None:
                best_dist = float(dists.loc[j])
                best_idx  = j
                best_df   = cand

        if best_idx is not None:
            out.at[i, "NEAR_DC_STOPNUMBER"]  = best_df.at[best_idx, "STOPNUMBER"]
            out.at[i, "NEAR_DC_CITY"]        = best_df.at[best_idx, "CITY"]
            out.at[i, "NEAR_DC_STATE"]       = best_df.at[best_idx, "STATE"]
            out.at[i, "NEAR_DC_PPG_NO_TAX"]  = float(best_df.at[best_idx, "PPG_NO_TAX"]) if pd.notna(best_df.at[best_idx, "PPG_NO_TAX"]) else None
            out.at[i, "NEAR_DC_DISTANCE_MI"] = best_dist

        # Choose event price: nearest DC within threshold, else WEX state median
        dc_price = out.at[i, "NEAR_DC_PPG_NO_TAX"]
        wex_fallback = out.at[i, "WEX_STATE_NET_EX_MEDIAN"]
        if dc_price is not None and (out.at[i, "NEAR_DC_DISTANCE_MI"] is None or out.at[i, "NEAR_DC_DISTANCE_MI"] <= nearest_within_miles):
            out.at[i, "EVENT_PPG"] = float(dc_price)
        else:
            out.at[i, "EVENT_PPG"] = float(wex_fallback) if pd.notna(wex_fallback) else None

    out["STATE_KEY"] = out["STATE"].map(_state_key)
    out["WEX_STATE_NET_EX_MEDIAN"] = out["STATE_KEY"].map(state_price)

    # only use a DC price when it's within the threshold
    mask_near_dc = (
        pd.notna(out["NEAR_DC_PPG_NO_TAX"]) &
        (out["NEAR_DC_DISTANCE_MI"].isna() | (out["NEAR_DC_DISTANCE_MI"] <= nearest_within_miles))
    )
    out["EVENT_PPG"]    = np.where(mask_near_dc, out["NEAR_DC_PPG_NO_TAX"], out["WEX_STATE_NET_EX_MEDIAN"])
    out["PRICE_SOURCE"] = np.where(mask_near_dc, "DC_NEARBY", "WEX_MEDIAN")
    return out

# ---- cell 31 ----
def prepare_events_for_miles(session, dot_events_sp):
    DOT_TBL    = "DEV_DOT_MLAI.DTI_FUEL.DTI_DOT_LOCATIONS"  # DOT_NAME, LAT, LON, GEOG
    STATES_TBL = "DEV_DOT_MLAI.DTI_FUEL.STATES_RAW"          # NAME, GEOG

    e = dot_events_sp.alias("e")
    d = session.table(DOT_TBL).alias("d")
    s = session.table(STATES_TBL).alias("s")

    # Be tolerant of either LOAD_NUMBER or LOAD_NO coming in
    e_cols = {c.upper(): c for c in e.columns}
    load_col     = e_cols.get("LOAD_NUMBER") or e_cols.get("LOAD_NO")
    from_seq_col = e_cols.get("FROM_SEQ") or "FROM_SEQ"
    to_seq_col   = e_cols.get("TO_SEQ")   or "TO_SEQ"
    dist_col     = e_cols.get("DIST_MI")  or "DIST_MI"

    if not load_col:
        raise ValueError(f"dot_events_sp is missing LOAD_NO/LOAD_NUMBER. Columns: {e.columns}")

    # Join in DOT geogs and state polygons (no string-literal SQL; use API to avoid quoting surprises)
    base = (
        e.join(d, e["DOT_NAME"] == d["DOT_NAME"], how="left")
         .join(s, F.call_function("ST_CONTAINS", s["GEOG"], d["GEOG"]), how="left")
         .with_column("LOAD_NO",    e[load_col])  # normalize for downstream pandas
         .with_column("LON",       d["LON"])
         .with_column("LAT",        d["LAT"])
         .with_column("STATE",      s["NAME"])
         .with_column("EVENT_TYPE", F.lit("DOT"))
         .with_column("IS_DC",      F.lit(False))
    )

    # Window for sequencing events per load
    w = Window.partition_by("LOAD_NO").order_by(
        F.col(from_seq_col), F.col(to_seq_col), F.col(dist_col)
    )

    out_sp = (
        base.with_column("EVENT_SEQ", F.row_number().over(w))
            .select(
                "LOAD_NO", "EVENT_SEQ", "EVENT_TYPE", "IS_DC",
                "STATE", "LON", "LAT",
                d["DOT_NAME"].alias("DOT_NAME"),
                F.col(dist_col).alias("DIST_MI"),
                F.col(from_seq_col).alias("FROM_SEQ"),
                F.col(to_seq_col).alias("TO_SEQ"),
            )
    )

    return out_sp.to_pandas()

# ---- cell 32 ----
def _state_key(s: str) -> str | None:
    if not s: return None
    s = s.strip()
    return STATE_TO_ABBR.get(s, s).upper()

# ---- cell 33 ----
def build_state_events_for_rcsp(state_segs_sp) -> "pd.DataFrame":
    """
    Turn state segments into a minimal events frame for RCSP:
      LOAD_NO, EVENT_SEQ, EVENT_TYPE='STATE', STATE, EVENT_MILES_TO_NEXT
    EVENT_MILES_TO_NEXT is set to the segment miles (distance to the next state boundary).
    """
    import pandas as pd

    segs = (
        state_segs_sp
        .select("LOAD_NUMBER","FROM_SEQ","TO_SEQ","STATE","SEG_MI_SCALED","START_IDX")
        .to_pandas()
        .sort_values(["LOAD_NUMBER","FROM_SEQ","START_IDX"])
        .reset_index(drop=True)
    )

    out = []
    for load_no, grp in segs.groupby("LOAD_NUMBER", sort=False):
        seq = 1
        for r in grp.itertuples(index=False):
            out.append({
                "LOAD_NO": int(load_no),
                "EVENT_SEQ": seq,
                "EVENT_TYPE": "STATE",
                "STATE": r.STATE,
                # RCSP uses this to compute miles-to-exit via cumulative miles
                "EVENT_MILES_TO_NEXT": float(r.SEG_MI_SCALED) if r.SEG_MI_SCALED is not None else 0.0,
                "FROM_SEQ": int(r.FROM_SEQ),
                "TO_SEQ": int(r.TO_SEQ),
                "LON": None,
                "LAT": None,
            })
            seq += 1

    return pd.DataFrame(out)


# ---- cell 34 ----
def annotate_miles_to_exit(df: "pd.DataFrame") -> "pd.DataFrame":
    # assumes df already sorted by LOAD_NO, EVENT_SEQ
    df = df.copy()
    df["HOS_MILES_TO_EXIT"] = None
    for load_no, sub in df.groupby("LOAD_NO", sort=False):
        types  = sub["EVENT_TYPE"].str.upper().tolist()
        states = sub["STATE"].astype(str).tolist()
        miles  = sub["EVENT_MILES_TO_NEXT"].fillna(0.0).astype(float).tolist()
        idxs   = sub.index.tolist()

        for pos, idx in enumerate(idxs):
            if types[pos] != "STATE":
                continue
            cur = states[pos]
            s = 0.0
            j = pos
            # Sum until the next STATE row with a different state (exclusive)
            while j < len(miles):
                s += miles[j]
                j += 1
                if j < len(miles) and types[j] == "STATE" and states[j] != cur:
                    break
            df.at[idx, "HOS_MILES_TO_EXIT"] = round(s, 3)
    return df

# ---- cell 35 ----
def annotate_rcsp_fee(
    events_df,
    load_no,
    *,
    default_fee: float = 100.0,
    plan_speed_mph: float = 50.0,
    start_drive_hours: float = 10.0,
    free_window_hours: float = 1.0,  # NEW
):
    import pandas as pd
    if events_df is None or len(events_df) == 0:
        return events_df

    df = events_df.copy()
    df = df.sort_values(["LOAD_NO","EVENT_SEQ"]).reset_index(drop=True)
    df["EVENT_MILES_TO_NEXT"] = pd.to_numeric(df.get("EVENT_MILES_TO_NEXT", 0.0), errors="coerce").fillna(0.0)

    if "HOS_MILES_TO_EXIT" not in df.columns or df["HOS_MILES_TO_EXIT"].isna().any():
        df = annotate_miles_to_exit(df)

    df["RCSP_PAY"]          = False
    df["RCSP_RATIONALE"]    = ""
    df["HOS_REM_DRIVE_H"]   = None
    df["HOS_AVG_SPEED_MPH"] = plan_speed_mph
    df["RCSP_FEE"]          = default_fee
    df["HOS_FREE_WINDOW_H"] = float(free_window_hours)  # audit aid

    t = df["EVENT_TYPE"].str.upper()
    eligible = t.eq("STATE") | t.eq("DC") | (t.eq("STOP") & df.get("IS_DC", False).fillna(False))
    hos = float(start_drive_hours)

    for i, r in df[df["LOAD_NO"] == load_no].iterrows():
        df.at[i, "HOS_REM_DRIVE_H"] = hos
        free_window = hos <= float(free_window_hours)
        if free_window:
            df.at[i, "RCSP_FEE"] = 0.0
            df.at[i, "RCSP_PAY"] = False
            df.at[i, "RCSP_RATIONALE"] = f"Free fuel window: remaining HOS ≤ {free_window_hours}h."
        else:
            if eligible.at[i]:
                df.at[i, "RCSP_FEE"] = float(default_fee)
                df.at[i, "RCSP_PAY"] = True
                if t.at[i] == "STATE":
                    miles_to_exit = df.at[i, "HOS_MILES_TO_EXIT"]
                    df.at[i, "RCSP_RATIONALE"] = f"State boundary decision; exit≈{miles_to_exit if miles_to_exit is not None else 'unknown'} mi."
                else:
                    df.at[i, "RCSP_RATIONALE"] = "Fuel-eligible DC/STOP."

        leg_mi = float(r["EVENT_MILES_TO_NEXT"] or 0.0)
        hos = max(0.0, hos - (leg_mi / plan_speed_mph))
        if hos <= 0.0:
            hos = 11.0  # reset
    return df


# ---- cell 36 ----
def build_unified_events(session, *, stops_sp, legs_sp, dc_events_pd, boundary_events_pd) -> "pd.DataFrame":
    # Build leg miles and cumulative stop mile-markers (unchanged)
    legs_pd = (
        legs_sp.select("LOAD_NUMBER", "FROM_SEQ", "TO_SEQ", "MILES_PCMILER")
               .to_pandas()
               .sort_values(["LOAD_NUMBER","FROM_SEQ"])
               .reset_index(drop=True)
    )
    legs_pd["LEG_MI"] = pd.to_numeric(legs_pd["MILES_PCMILER"], errors="coerce").fillna(0.0)

    mm_at_stop = {}
    for load_no, g in legs_pd.groupby("LOAD_NUMBER", sort=False):
        g = g.sort_values("FROM_SEQ")
        mm = {0: 0.0}
        for r in g.itertuples(index=False):
            prev = int(r.FROM_SEQ); nxt = int(r.TO_SEQ)
            base = mm.get(prev, 0.0)
            mm[nxt] = base + float(r.LEG_MI or 0.0)
        mm_at_stop[int(load_no)] = mm

    st_pd = (
        stops_sp.select("LOAD_NUMBER", "LOAD_STOP_SEQUENCE_NO", "STATE", "LON", "LAT", "DOT_DOMICILEABBREVIATION")
                .to_pandas()
                .sort_values(["LOAD_NUMBER", "LOAD_STOP_SEQUENCE_NO"])
                .reset_index(drop=True)
    )

    def _mm_for_stop(row):
        m = mm_at_stop.get(int(row["LOAD_NUMBER"]), {})
        return float(m.get(int(row["LOAD_STOP_SEQUENCE_NO"]), 0.0))

    st_pd["MM_APPROX_MI"] = st_pd.apply(_mm_for_stop, axis=1)

    stop_ev = st_pd.rename(columns={"LOAD_NUMBER": "LOAD_NO"})
    stop_ev["FROM_SEQ"]   = stop_ev["LOAD_STOP_SEQUENCE_NO"] - 1
    stop_ev["TO_SEQ"]     = stop_ev["LOAD_STOP_SEQUENCE_NO"]
    stop_ev["EVENT_TYPE"] = "STOP"
    stop_ev["IS_DC"]      = stop_ev["DOT_DOMICILEABBREVIATION"].fillna("").astype(str).str.len().gt(0)
    stop_ev["IS_STATE_BOUNDARY"] = False
    stop_ev["DOT_NAME"] = None
    stop_ev = stop_ev[[
        "LOAD_NO","FROM_SEQ","TO_SEQ","STATE","LON","LAT",
        "MM_APPROX_MI","EVENT_TYPE","IS_DC","IS_STATE_BOUNDARY","DOT_NAME"
    ]]

    # Origin STATE event
    origin_state_df = (
        st_pd.sort_values(["LOAD_NUMBER", "LOAD_STOP_SEQUENCE_NO"])
             .groupby("LOAD_NUMBER", as_index=False).head(1)
             .assign(
                 FROM_SEQ=-1, TO_SEQ=0,
                 MM_APPROX_MI=0.0,
                 EVENT_TYPE="STATE",
                 IS_STATE_BOUNDARY=False,
                 IS_DC=False
             )
             .rename(columns={"LOAD_NUMBER": "LOAD_NO"})
             [["LOAD_NO", "FROM_SEQ", "TO_SEQ", "STATE", "LON", "LAT",
               "MM_APPROX_MI", "EVENT_TYPE", "IS_DC", "IS_STATE_BOUNDARY"]]
    )

    # DC events (per-leg mm) -> globalize by adding leg base offset
    dc_ev = dc_events_pd.copy()
    dc_ev["IS_STATE_BOUNDARY"] = False
    if not dc_ev.empty:
        dc_ev["LEG_BASE_MM"] = dc_ev.apply(lambda r: mm_at_stop.get(int(r["LOAD_NO"]), {}).get(int(r["FROM_SEQ"]), 0.0), axis=1)
        dc_ev["MM_APPROX_MI"] = pd.to_numeric(dc_ev["MM_APPROX_MI"], errors="coerce").fillna(0.0) + dc_ev["LEG_BASE_MM"]
        dc_ev = dc_ev.drop(columns=["LEG_BASE_MM"])
    dc_ev = dc_ev[["LOAD_NO", "FROM_SEQ", "TO_SEQ", "STATE", "LON", "LAT",
                   "MM_APPROX_MI", "EVENT_TYPE", "IS_DC", "IS_STATE_BOUNDARY", "DOT_NAME"]]

    # State boundary events (per-leg mm) -> globalize by adding leg base offset
    b_ev = boundary_events_pd.copy()
    b_ev["IS_DC"] = False
    b_ev["DOT_NAME"] = None
    if not b_ev.empty:
        b_ev["LEG_BASE_MM"] = b_ev.apply(lambda r: mm_at_stop.get(int(r["LOAD_NO"]), {}).get(int(r["FROM_SEQ"]), 0.0), axis=1)
        b_ev["MM_APPROX_MI"] = pd.to_numeric(b_ev["MM_APPROX_MI"], errors="coerce").fillna(0.0) + b_ev["LEG_BASE_MM"]
        b_ev = b_ev.drop(columns=["LEG_BASE_MM"])
    b_ev = b_ev[["LOAD_NO", "FROM_SEQ", "TO_SEQ", "STATE", "LON", "LAT",
                 "MM_APPROX_MI", "EVENT_TYPE", "IS_DC", "IS_STATE_BOUNDARY", "DOT_NAME"]]

    ev = pd.concat([stop_ev, origin_state_df.assign(DOT_NAME=None), b_ev, dc_ev], ignore_index=True)

    # Normalize states to two-letter codes
    ev["STATE"] = ev["STATE"].apply(lambda s: _state_key(s) or s)

    # Dedup/origin ordering (as you had)
    mask_stop_is_dc0 = ev["EVENT_TYPE"].eq("STOP") & ev["IS_DC"].fillna(False) & ev["MM_APPROX_MI"].abs().le(1e-6)
    ev.loc[mask_stop_is_dc0, "EVENT_TYPE"] = "DC"

    mm0 = ev["MM_APPROX_MI"].abs().le(1e-6)
    dup_key = (
        ev["LOAD_NO"].astype(str) + "|" +
        ev["EVENT_TYPE"].astype(str) + "|" +
        ev["LON"].round(6).astype(str) + "|" +
        ev["LAT"].round(6).astype(str) + "|" +
        ev["MM_APPROX_MI"].round(3).astype(str)
    )
    ev = ev.loc[~(mm0 & dup_key.duplicated(keep="first"))].copy()

    rank = {"STOP": 0, "DC": 0, "STATE": 1}
    ev["__r"] = ev["EVENT_TYPE"].str.upper().map(rank).fillna(9)
    ev = ev.sort_values(["LOAD_NO", "MM_APPROX_MI", "__r"]).drop(columns="__r").reset_index(drop=True)

    ev["EVENT_SEQ"] = ev.groupby("LOAD_NO", sort=False).cumcount() + 1
    return ev


# ---- cell 37 ----
def optimize_fuel_plan_rcsp(
        events,                # list[dict] in route order
        initial_fuel,          # gallons on board at first event
        mpg,                   # truck MPG
        tank_capacity,         # max gallons
        safety_buffer,         # reserve gallons that must remain after each leg
        delta: float = 0.1,    # purchase “click” size in gallons
        stop_fee: float = 100.0,  # fee if a purchase happens and RCSP_FEE absent
):
    """
    Label-setting RCSP on a 3-D state space:
      state = (i, u, paid) where
        i     : event index
        u     : fuel units on board (integerised gallons / delta)
        paid  : 0 if stop fee not yet paid at node i, 1 if already paid at i

    Returns: list of (EVENT_SEQ, gallons_bought) tuples aggregated per stop.
    """
    cons_per_mile = 1.0 / float(mpg)
    N = len(events)
    max_units = int(float(tank_capacity) / delta)
    buf_units = int(float(safety_buffer) / delta)

    # miles needed (in fuel units) to reach i+1
    leg_req = [
        int(math.ceil(((ev.get("EVENT_MILES_TO_NEXT") or 0.0) * cons_per_mile) / delta))
        for ev in events
    ]

    INF = float("inf")
    # dist[paid][i][u] = min cost to be at (i,u,paid)
    dist = [[[INF]*(max_units + 1) for _ in range(N)] for _ in range(2)]
    prev = [[[None]*(max_units + 1) for _ in range(N)] for _ in range(2)]

    start_u = min(int(float(initial_fuel) / delta), max_units)
    dist[1][0][start_u] = 0.0                 # at origin; treat fee as already paid
    heap = [(0.0, 0, start_u, 1)]             # (cost, i, u, paid)

    while heap:
        cost, i, u, paid = heapq.heappop(heap)
        if cost > dist[paid][i][u]:
            continue

        # Drive to next node
        if i + 1 < N:
            req = leg_req[i]
            if req == 0:
                # Zero-mile hop: allow advancing without enforcing the buffer
                v = u
                if cost < dist[0][i+1][v]:
                    dist[0][i+1][v] = cost
                    prev[0][i+1][v] = (paid, i, u, 0.0)
                    heapq.heappush(heap, (cost, i+1, v, 0))
            elif u - req >= buf_units:
                # Real driving leg: enforce reserve
                v = u - req
                if cost < dist[0][i+1][v]:
                    dist[0][i+1][v] = cost
                    prev[0][i+1][v] = (paid, i, u, 0.0)
                    heapq.heappush(heap, (cost, i+1, v, 0))

        # Buy one delta gallon at node i
        price = events[i].get("PRICE")
        if price is not None and u < max_units:
            v = u + 1
            node_fee = events[i].get("RCSP_FEE", stop_fee)
            fee_inc  = 0.0 if paid else float(node_fee)
            new_c    = cost + fee_inc + float(price) * delta
            if new_c < dist[1][i][v]:
                dist[1][i][v] = new_c
                prev[1][i][v] = (paid, i, u, delta)
                heapq.heappush(heap, (new_c, i, v, 1))

    # Best terminal state
    end_i = N - 1
    best_u, best_paid, best_c = None, None, INF
    for p in (0, 1):
        for u in range(buf_units, max_units + 1):
            if dist[p][end_i][u] < best_c:
                best_c, best_u, best_paid = dist[p][end_i][u], u, p
    if best_u is None:
        raise RuntimeError("No feasible refuelling plan found.")

    # Backtrack purchases → aggregate by EVENT_SEQ
    plan = []
    cur = (best_paid, end_i, best_u)
    while True:
        p, i, u = cur
        prv = prev[p][i][u]
        if prv is None:
            break
        pp, pi, pu, bought = prv
        if bought > 0:
            plan.append((events[i]["EVENT_SEQ"], bought))
        cur = (pp, pi, pu)

    plan.reverse()
    agg = {}
    for seq, g in plan:
        agg[int(seq)] = round(agg.get(int(seq), 0.0) + float(g), 1)
    return [(seq, gals) for seq, gals in agg.items()]

# ---- cell 38 ----
def build_state_boundary_events_from_geotunnel(routes_with_geotunnel_pd) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for r in routes_with_geotunnel_pd.itertuples(index=False):
        load_no = getattr(r, "LOAD_NO")
        from_seq = getattr(r, "LOAD_STOP_SEQUENCE_NO")
        to_seq = from_seq + 1
        for t in getattr(r, "STATE_TRANSITIONS", None) or []:
            rows.append({
                "LOAD_NO": int(load_no),
                "FROM_SEQ": int(from_seq),
                "TO_SEQ": int(to_seq),
                "STATE": t.get("to_state"),
                "LON": float(t.get("transition_lon")) if t.get("transition_lon") is not None else None,
                "LAT": float(t.get("transition_lat")) if t.get("transition_lat") is not None else None,
                "MM_APPROX_MI": float(t.get("mm")) if t.get("mm") is not None else None,
                "EVENT_TYPE": "STATE",
                "IS_STATE_BOUNDARY": True,
            })
    return pd.DataFrame(rows, columns=[
        "LOAD_NO","FROM_SEQ","TO_SEQ","STATE","LON","LAT","MM_APPROX_MI",
        "EVENT_TYPE","IS_STATE_BOUNDARY"
    ])


# ---- cell 39 ----
def _polyline_length_mi(points):
    """Accurate haversine length of a polyline [(lon,lat), ...] in miles."""
    if not points: 
        return 0.0
    from math import radians, sin, cos, asin, sqrt
    R = 3958.7613
    total = 0.0
    lon0, lat0 = points[0]
    for lon1, lat1 in points[1:]:
        dlat = radians(lat1 - lat0); dlon = radians(lon1 - lon0)
        a = sin(dlat/2)**2 + cos(radians(lat0))*cos(radians(lat1))*sin(dlon/2)**2
        total += 2 * R * asin(sqrt(a))
        lon0, lat0 = lon1, lat1
    return total


# ---- cell 40 ----
def _format_plan_text(events_df, plan_list, params):
    rows = []
    total_cost = 0.0
    by_seq = events_df.set_index("EVENT_SEQ")
    for seq, gals in plan_list:
        r = by_seq.loc[int(seq)]
        et = str(r.get("EVENT_TYPE") or "").upper()
        is_dc = bool(r.get("IS_DC"))
        if et in ("DC", "STOP") and is_dc:
            name = r.get("DOT_NAME")
            if not name:
                near_city  = r.get("NEAR_DC_CITY")
                near_state = r.get("NEAR_DC_STATE") or r.get("STATE")
                name = f"{near_city}, {near_state}" if near_city else str(near_state or "DC")
            loc = f"DC {name}"
        elif et == "STATE":
            loc = f"State line: {r.get('STATE')}"
        else:
            loc = f"Stop #{int(r.get('TO_SEQ') or r.get('FROM_SEQ') or 0)}"

        ppg  = float(r.get("PRICE") or 0.0)
        cost = round(gals * ppg, 2) if ppg else None
        if cost is not None:
            total_cost += cost
        rows.append(f"- Buy **{gals:.1f} gal** at {loc}" + (f" @ ${ppg:.3f}/gal (≈${cost:.2f})" if ppg else ""))

    header = (f"Load {int(params['load'])}: Start {params['initial']} gal in {int(params['tank'])} gal tank "
              f"(MPG {params['mpg']}, reserve {params['buffer']} gal).")
    footer = f"Estimated fuel spend: ${total_cost:.2f}" if total_cost else ""
    return "\n".join([header, "", *rows, "", footer]).strip()


# ---- cell 41 ----
def get_llm_plan(events_all, plan_list, params: dict) -> str:
    """
    Use GPT-5-mini to render a clean, readable Markdown fuel plan.
    Falls back to a simple template if the API fails.
    """
    try:
        buy_map = {int(seq): float(gal) for (seq, gal) in plan_list}
        view_cols = [
            "EVENT_SEQ","EVENT_TYPE","LOCATION_TYPE","STATE","DOT_NAME",
            "NEAR_DC_CITY","NEAR_DC_STATE","EVENT_PPG","PRICE",
            "EVENT_MILES_TO_NEXT","GALLONS_TO_NEXT"
        ]
        df = events_all.copy()
        for c in view_cols:
            if c not in df.columns:
                df[c] = np.nan
        df["BUY_GALLONS"] = df["EVENT_SEQ"].map(buy_map).fillna(0.0)

        payload = {
            "meta": {
                "load_number": params.get("load"),
                "initial_fuel_gal": float(params.get("initial", 0)),
                "tank_capacity_gal": float(params.get("tank", 0)),
                "mpg": float(params.get("mpg", 0)),
                "reserve_gal": float(params.get("buffer", 0)),
            },
            "events": df[view_cols + ["BUY_GALLONS"]].replace({np.nan: None}).to_dict("records"),
        }

        system = (
            "You format trucking fuel plans. Output concise Markdown only. "
            "Use the JSON exactly—no invented values. Include: title; quick summary; "
            "itemized buys (location/state, event_seq, price, gallons, est cost); "
            "compact table of all hops (seq, type, loc/state, mi_to_next, gal_to_next, price); "
            "and totals (gallons bought, spend). Money to 2 decimals."
        )
        user = f"Render this fuel plan nicely. JSON:\n\n{payload}"

        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception:
        buys = [f"- Seq {int(seq)}: buy **{float(g):.1f} gal**" for (seq, g) in plan_list]
        return (
            f"Load {params.get('load')}: start {params.get('initial')} gal in {params.get('tank')} gal tank "
            f"(MPG {params.get('mpg')}, reserve {params.get('buffer')} gal).\n\n" + "\n".join(buys)
        )

# ---- cell 42 ----
def render_compact_fuel_plan(events_df, plan_rows, title_prefix=None):
    """
    events_df: DataFrame with EVENT_SEQ, EVENT_TYPE, STATE, DOT_NAME, PRICE, EVENT_MILES_TO_NEXT
               (optional: NEAR_DC_CITY, NEAR_DC_STATE, CITY)
    plan_rows: list of dicts like {'EVENT_SEQ': int, 'BUY_GALLONS': float}
    """
    import math
    import numpy as np
    import pandas as pd

    header = "Fuel Plan" if not title_prefix else f"{title_prefix}\n\nFuel Plan"
    if events_df is None or len(events_df) == 0:
        return f"No fuel purchases.\n\n{header}\n"

    # Prepare and compute fallback leg miles (use next event's miles if current is 0)
    df = events_df.copy().sort_values("EVENT_SEQ").reset_index(drop=True)
    df["EVENT_MILES_TO_NEXT"] = pd.to_numeric(df.get("EVENT_MILES_TO_NEXT"), errors="coerce").fillna(0.0)
    miles = df["EVENT_MILES_TO_NEXT"].to_numpy()
    miles_next = np.r_[miles[1:], 0.0]
    df["LEG_MILES_FOR_LINE"] = np.where(miles > 0, miles, miles_next)
    leg_mi_map = dict(zip(df["EVENT_SEQ"].astype(int), df["LEG_MILES_FOR_LINE"]))

    by_seq = df.set_index("EVENT_SEQ", drop=False)

    def _uc(x):  # uppercase safe
        return str(x or "").strip().upper()

    def label_for(seq: int) -> str:
        r = by_seq.loc[int(seq)]
        et = _uc(r.get("EVENT_TYPE"))
        state_abbr = _uc(r.get("STATE"))
        dot = (r.get("DOT_NAME") or "").strip()

        if et == "DC":
            city = (r.get("NEAR_DC_CITY") or (dot.split(",")[0].strip() if dot else "")).title()
            st = _uc(r.get("NEAR_DC_STATE") or state_abbr)
            name = ", ".join([p for p in [city, st] if p])
            return f"{name} DC" if name else "DC"

        if et == "STATE":
            prev_state = _uc(by_seq.loc[int(seq) - 1]["STATE"]) if (int(seq) - 1) in by_seq.index else ""
            if prev_state and prev_state != state_abbr:
                return f"{state_abbr} (entering from {prev_state})"
            return state_abbr or "STATE"

        # Default: try City, ST; otherwise just ST
        city = (r.get("CITY") or "").title()
        return (f"{city}, {state_abbr}".strip(", ") if city else state_abbr) or "Location"

    items, lines = [], []

    for i, row in enumerate(plan_rows, start=1):
        seq = int(row["EVENT_SEQ"])
        buy = float(row["BUY_GALLONS"])
        r = by_seq.loc[seq]
    
        lab = label_for(seq)
        leg_mi = int(round(float(leg_mi_map.get(seq, 0.0))))
        items.append(f"buy {buy:.1f} gallons at {lab}")
    
        # robust price handling (no $0.000 when missing)
        raw_price = r.get("PRICE")
        try:
            price = float(raw_price)
            if math.isnan(price):
                price = None
        except (TypeError, ValueError):
            price = None
    
        price_str = f"${price:.3f}/gal" if price is not None else "N/A"
        cost_str  = f"${buy*price:.0f}" if price is not None else "N/A"
    
        lines.append(
            f"{i}. {lab}: buy {buy:.1f} gal @ {price_str} = {cost_str}"
            + (f"; next leg {leg_mi} mi" if leg_mi > 0 else "")
        )

    # summary sentence
    if items:
        if len(items) > 1:
            summary = ", then ".join(items[:-1]) + f", and lastly {items[-1]}."
        else:
            summary = items[0][:1].upper() + items[0][1:] + "."
    else:
        summary = "No fuel purchases."


    return f"{summary}\n\n{header}\n" + "\n".join(lines)


# ---- cell 43 ----
def main(session: Session, inputs) -> dict:
    run_id = getattr(inputs, "run_id", uuid.uuid4().hex)
    stops = inputs.stops_df
    legs_sp = build_legs_and_compute_mileage(session=session, stops_df=stops, load_number=inputs.load_number)

    routes_with_geotunnel = build_routes_df_for_events(session, stops, legs_sp, pcmiler_api_key=PC_MILER_API_KEY)
    boundary_events_pd = build_state_boundary_events_from_geotunnel(routes_with_geotunnel)
    dc_events_pd = expand_dc_candidates_from_geotunnel(
        session,
        routes_with_geotunnel,
        radius_mi=getattr(inputs, "dc_radius_mi", 5.0),
        step_mi=getattr(inputs, "geotunnel_step_mi", 1.0),
    )

    events_all = build_unified_events(
        session,
        stops_sp=stops,
        legs_sp=legs_sp,
        dc_events_pd=dc_events_pd,
        boundary_events_pd=boundary_events_pd,
    ).reset_index(drop=True)

    mileage_fn = create_mileage_cache(PC_MILER_API_KEY)
    events_all = add_event_miles(events_all, mileage_fn)

    mm_next = events_all["MM_APPROX_MI"].groupby(events_all["LOAD_NO"]).shift(-1)
    hop_mm  = (mm_next - events_all["MM_APPROX_MI"]).clip(lower=0).fillna(0.0)
    events_all["EVENT_MILES_TO_NEXT"] = np.where(
        events_all["EVENT_MILES_TO_NEXT"].notna() & (events_all["EVENT_MILES_TO_NEXT"] > 0),
        events_all["EVENT_MILES_TO_NEXT"],
        hop_mm,
    )
    events_all["EVENT_MILES_TO_NEXT"] = (
        pd.to_numeric(events_all["EVENT_MILES_TO_NEXT"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    # NEW: harmonize Σ hops with final mile-marker
    events_all = reconcile_event_miles(events_all, tol_mi=0.5)

    events_all = annotate_miles_to_exit(events_all)

    dc_fuel_df, state_price = build_price_inputs(session)
    events_all = append_price_info(events_all, dc_fuel_df, state_price)

    is_state = events_all["EVENT_TYPE"].str.upper().eq("STATE")
    is_dc    = events_all["EVENT_TYPE"].str.upper().eq("DC") | (
                 events_all["EVENT_TYPE"].str.upper().eq("STOP") & events_all["IS_DC"].fillna(False)
               )
    eligible = (is_state | is_dc)
    events_all["PRICE"] = np.where(eligible, events_all["EVENT_PPG"], np.nan)
    
    # Origin feasibility override (pre-trip top-off)
    events_all = events_all.sort_values(["LOAD_NO","EVENT_SEQ"]).reset_index(drop=True)
    # Find the first hop that actually consumes miles
    next_pos = events_all[events_all["EVENT_MILES_TO_NEXT"] > 0].head(1)
    origin_needs_buy = False
    if not next_pos.empty:
        need_gal_first_move = float(next_pos.iloc[0]["EVENT_MILES_TO_NEXT"]) / float(inputs.mpg)
        origin_needs_buy = (float(inputs.initial_fuel) - need_gal_first_move) < float(inputs.safety_buffer)
    
    # If we *do* need to buy before that first consuming hop, make the origin eligible/priced
    if origin_needs_buy:
        # ensure the very first event has a usable price (prefer nearby DC within 5 mi)
        first = events_all.iloc[0]
        dc_price = first.get("NEAR_DC_PPG_NO_TAX")
        dist     = first.get("NEAR_DC_DISTANCE_MI")
        chosen   = dc_price if (pd.notna(dc_price) and (pd.isna(dist) or float(dist) <= 5.0)) else first.get("EVENT_PPG")
        events_all.at[0, "PRICE"] = chosen
        events_all.at[0, "IS_DC"] = True


    if "PRICE" not in events_all.columns:
        events_all["PRICE"] = events_all["EVENT_PPG"]

    multiple_drivers = is_team_driving(stops)
    
    if multiple_drivers:
        # HARD BYPASS: don't call annotate_rcsp_fee at all
        events_all["HOS_APPLIED"]       = False
        events_all["HOS_REM_DRIVE_H"]   = np.nan
        events_all["HOS_AVG_SPEED_MPH"] = np.nan
        events_all["HOS_FREE_WINDOW_H"] = 0.0
        events_all["HOS_MILES_TO_EXIT"] = np.nan
        # keep fee model simple when team driving
        events_all["RCSP_FEE"]          = 100.0
        events_all["RCSP_PAY"]          = True
    else:
        start_drive_h = get_start_drive_hours_for_load(session, stops)
        events_all = annotate_rcsp_fee(
        events_all,
        load_no=inputs.load_number,
        default_fee=100.0,
        plan_speed_mph=50.0,
        start_drive_hours=float(start_drive_h),
        free_window_hours=1.0,
        )
        events_all["HOS_APPLIED"] = True

    # ---- fee-free if HOS hits 0 inside the state segment ----
    ev = events_all.copy()
    
    is_state = ev["EVENT_TYPE"].str.upper().eq("STATE")
    ev["SEGMENT_ID"] = is_state.cumsum()
    
    mph    = ev["HOS_AVG_SPEED_MPH"].fillna(50.0).astype(float)
    hos_h = pd.to_numeric(ev["HOS_REM_DRIVE_H"], errors="coerce").fillna(0.0)
    seg_mi = pd.to_numeric(
        ev["HOS_MILES_TO_EXIT"].where(is_state).fillna(ev["EVENT_MILES_TO_NEXT"]),
        errors="coerce"
    ).fillna(0.0)
    
    hits_zero_at_state = (mph * hos_h) <= seg_mi
    ev["SEG_FEE_FREE"] = False
    ev.loc[is_state & hits_zero_at_state, "SEG_FEE_FREE"] = True
    ev["SEG_FEE_FREE"] = ev.groupby("SEGMENT_ID")["SEG_FEE_FREE"].transform("max")
    
    fee_eligible = ev["EVENT_TYPE"].str.upper().isin(["STATE","DC","STOP"])
    ev.loc[fee_eligible & ev["SEG_FEE_FREE"], "RCSP_FEE"] = 0.0
    ev.loc[fee_eligible & ev["SEG_FEE_FREE"], "RCSP_PAY"] = False
    ev.loc[fee_eligible & ev["SEG_FEE_FREE"], "RCSP_RATIONALE"] = (
        ev.loc[fee_eligible & ev["SEG_FEE_FREE"], "RCSP_RATIONALE"].fillna("").str.rstrip()
        + " HOS hits 0 in this state segment; fee waived."
    )
    
    events_all = ev


    # Optimizer inputs
    truck_mpg = float(inputs.mpg)
    events_all["LOCATION_TYPE"] = np.select(
        [
            (events_all["EVENT_TYPE"].str.upper() == "STOP") & events_all["IS_DC"].fillna(False),
            (events_all["EVENT_TYPE"].str.upper() == "STOP") & (~events_all["IS_DC"].fillna(False)),
            (events_all["EVENT_TYPE"].str.upper() == "STATE"),
            (events_all["EVENT_TYPE"].str.upper() == "DC"),
        ],
        ["DC", "STOP", "STATE", "DC"],
        default="SKIP"
    )
    events_all["GALLONS_TO_NEXT"] = (events_all["EVENT_MILES_TO_NEXT"] / truck_mpg).round(2)

    plan_list = optimize_fuel_plan_rcsp(
        events_all.to_dict(orient="records"),
        initial_fuel=float(inputs.initial_fuel),
        mpg=truck_mpg,
        tank_capacity=float(inputs.tank_capacity),
        safety_buffer=float(inputs.safety_buffer),
        stop_fee=100.0,
    )

    # Previews/telemetry (optional)
    geotunnel_preview = routes_with_geotunnel.head(50).to_dict(orient="records")
    dc_events_preview = dc_events_pd.head(100).to_dict(orient="records")
    boundary_preview  = boundary_events_pd.head(100).to_dict(orient="records")
    combined_preview  = events_all.head(50).to_dict(orient="records")
    plan_preview      = [{"EVENT_SEQ": int(seq), "BUY_GALLONS": float(g)} for (seq, g) in plan_list]

    # fuel_plan_text = _format_plan_text(
    # events_all,
    # plan_list,
    # params={"load": inputs.load_number, "initial": inputs.initial_fuel,
    #             "tank": inputs.tank_capacity, "mpg": inputs.mpg, "buffer": inputs.safety_buffer}
    # )

    # fuel_plan_text = get_llm_plan(
    #     events_all,
    #     plan_list,
    #     params={"load": inputs.load_number, "initial": inputs.initial_fuel,
    #             "tank": inputs.tank_capacity, "mpg": inputs.mpg, "buffer": inputs.safety_buffer}
    # )

    events_df = events_all.reset_index(drop=True)          # the unified events you just built
    fuel_plan_text = render_compact_fuel_plan(events_df,   # pass the events
                                              plan_preview) 

    hos_cols = [c for c in events_df.columns if c.upper().startswith("HOS_")]
    safe_events_preview = (
        events_df.drop(columns=[c for c in hos_cols if c in events_df.columns], errors="ignore")
                 .head(200)
                 .to_dict(orient="records")
    )

    # --- JSON blobs ---
    payload_json = json.dumps(getattr(inputs, "raw_payload", {}))
    fuel_params_json = json.dumps({
        "initial_fuel": inputs.initial_fuel,
        "tank_capacity": inputs.tank_capacity,
        "mpg": inputs.mpg,
        "safety_buffer": inputs.safety_buffer,
    })
    plan_json   = json.dumps(plan_preview)
    events_json = json.dumps(safe_events_preview)
    dc_json     = json.dumps(dc_events_preview)
    state_json = boundary_events_pd.head(100).to_json(orient="records")
    
    # --- header row -> FUEL_RUN_LOG (10 cols incl CREATED_AT) ---
    tbl = "DEV_DOT_MLAI.DTI_FUEL.FUEL_RUN_LOG"
    
    vals = session.create_dataframe(
        [(run_id, str(inputs.load_number), payload_json, fuel_params_json, plan_json,
          fuel_plan_text, events_json, dc_json, state_json)],
        schema=[
            "RUN_ID","LOAD_NUMBER","REQUEST_PAYLOAD","FUEL_PARAMS","FUEL_PLAN",
            "FUEL_PLAN_TEXT","EVENTS_PREVIEW","DC_EVENTS_PREVIEW","STATE_BOUNDARY_PREVIEW",
        ],
    )
    
    tgt = session.table(tbl)
    provided = {
        "RUN_ID": F.col("RUN_ID"),
        "LOAD_NUMBER": F.col("LOAD_NUMBER"),
        "REQUEST_PAYLOAD": F.parse_json(F.col("REQUEST_PAYLOAD")),
        "FUEL_PARAMS": F.parse_json(F.col("FUEL_PARAMS")),
        "FUEL_PLAN": F.parse_json(F.col("FUEL_PLAN")),
        "FUEL_PLAN_TEXT": F.col("FUEL_PLAN_TEXT"),
        "EVENTS_PREVIEW": F.parse_json(F.col("EVENTS_PREVIEW")),
        "DC_EVENTS_PREVIEW": F.parse_json(F.col("DC_EVENTS_PREVIEW")),
        "STATE_BOUNDARY_PREVIEW": F.parse_json(F.col("STATE_BOUNDARY_PREVIEW")),
    }
    
    select_list = []
    for f in tgt.schema.fields:
        n = f.name.upper()
        if n == "CREATED_AT":
            expr = F.current_timestamp().alias(n)            # fill CREATED_AT (table default won’t apply on DF insert)
        else:
            expr = provided.get(n, F.lit(None).cast(f.datatype)).alias(n)
        select_list.append(expr)
    
    vals.select(select_list).create_or_replace_temp_view("TMP_LOG")
    session.sql(f'INSERT INTO {tbl} SELECT * FROM TMP_LOG').collect()
    
    # --- normalized buys -> FUEL_RUN_BUYS ---
    insert_df = session.create_dataframe(plan_preview).select(
        F.lit(run_id).cast(StringType()).alias("RUN_ID"),
        F.col("EVENT_SEQ").cast(IntegerType()).alias("EVENT_SEQ"),
        F.col("BUY_GALLONS").cast(DecimalType(10, 2)).alias("BUY_GALLONS"),
    )
    
    insert_df.create_or_replace_temp_view("TMP_BUYS")
    session.sql("""
      INSERT INTO DEV_DOT_MLAI.DTI_FUEL.FUEL_RUN_BUYS (RUN_ID, EVENT_SEQ, BUY_GALLONS)
      SELECT RUN_ID, EVENT_SEQ, BUY_GALLONS FROM TMP_BUYS
    """).collect()

    return {
        "status": "ok",
        "load_number": inputs.load_number,
        "fuel_plan": plan_preview,
        "fuel_plan_text": fuel_plan_text,
        "rows_in_payload": stops.count(),
        "fuel_params": {
            "initial_fuel": inputs.initial_fuel,
            "tank_capacity": inputs.tank_capacity,
            "mpg": inputs.mpg,
            "safety_buffer": inputs.safety_buffer
        },
        "dc_events_preview": dc_events_preview,
        "state_boundary_preview": boundary_preview,
        "events_unified_preview": combined_preview
    }


# ---- cell 44 ----
def validate_inputs(_inputs):
    print(_inputs)
    df = _inputs.stops_df
    print("== schema =="); print(df.schema)
    print("== head ==");  df.show(5)

def build_inputs_from_payload(payload: Any, session: Session, *, run_id: str | None = None) -> FuelPlanInputs:
    run = _coerce_payload_to_dict(payload)
    rid = run_id or run.get("RUN_ID") or uuid.uuid4().hex

    load_number   = get_num(run.get("LOAD_NUMBER"), int, None)
    initial_fuel  = get_num(run.get("INITIAL_FUEL"), float, None)
    tank_capacity = get_num(run.get("TANK_CAPACITY"), float, None)
    mpg           = get_num(run.get("MPG"), float, None)
    safety_buffer = get_num(run.get("SAFETY_BUFFER"), float, None)

    missing = [k for k, v in {
        "LOAD_NUMBER": load_number,
        "INITIAL_FUEL": initial_fuel,
        "TANK_CAPACITY": tank_capacity,
        "MPG": mpg,
        "SAFETY_BUFFER": safety_buffer,
    }.items() if v is None]
    if missing:
        raise ValueError(f"run payload missing required field(s): {', '.join(missing)}")

    stops_in = run.get("STOPS")
    if not isinstance(stops_in, list) or not stops_in:
        raise ValueError("run payload must contain a non-empty STOPS array.")
    if len(stops_in) < 2:
        raise ValueError("STOPS must contain at least an origin and a destination.")

    norm = []
    for s in stops_in:
        seq = get_num(s.get("LOAD_STOP_SEQUENCE_NO"), int, 0)
        norm.append({
            "LOAD_NUMBER":        load_number,
            "LOAD_STOP_SEQUENCE_NO": seq,
            "APPT_ETA_TS":        parse_ts(s.get("LOAD_STOP_SCHEDULEDSTARTDATE")),
            "STOP_NAME":          (s.get("DOT_DOMICILENAME") or s.get("STOP_NAME") or ""),
            "ADDRESS1":           (s.get("ADDRESSONE") or s.get("ADDRESS1") or ""),
            "CITY":               (s.get("CITY") or ""),
            "STATE":              (s.get("STATE") or ""),
            "POSTALCODE":         (s.get("POSTALCODE") or ""),
            "LON":                get_num(s.get("LON"), float, None),
            "LAT":                get_num(s.get("LAT"), float, None),
            "DRIVER_ONE_EMP_NO":  get_num(s.get("DRIVER_ONE_EMPLOYEE_NUMBER"), int, 0),
            "DRIVER_TWO_EMP_NO":  get_num(s.get("DRIVER_TWO_EMPLOYEE_NUMBER"), int, 0),
            "DOT_DOMICILEABBREVIATION": (s.get("DOT_DOMICILEABBREVIATION") or ""),
        })

    norm.sort(key=lambda r: r["LOAD_STOP_SEQUENCE_NO"])
    stops_df = session.create_dataframe(norm, schema=STOP_SCHEMA)

    return FuelPlanInputs(
        load_number=load_number,
        initial_fuel=initial_fuel,
        tank_capacity=tank_capacity,
        mpg=mpg,
        safety_buffer=safety_buffer,
        stops_df=stops_df,
        raw_payload=run,
        run_id=rid,
    )


def run_fuel_model(*, run_payload: Any | None = None, run_id: str | None = None) -> dict:
    payload = run_payload
    if payload is None and run_id:
        payload = _get_run_payload([f"RUN_ID={run_id}"], session)
    if payload is None:
        raise ValueError("Must provide run_payload or run_id")

    inputs = build_inputs_from_payload(payload, session, run_id=run_id)
    return main(session, inputs)


