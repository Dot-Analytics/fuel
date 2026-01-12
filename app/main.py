# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.fuel_model import run_fuel_model

app = FastAPI(title="DTI Fuel Model API")


class FuelRunRequest(BaseModel):
    run_id: str | None = Field(None, description="Existing RUN_ID to pull from Snowflake inbox")
    run_payload: dict | None = Field(None, description="Payload matching notebook RUN_JSON")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_model(req: FuelRunRequest):
    try:
        result = run_fuel_model(run_payload=req.run_payload, run_id=req.run_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

