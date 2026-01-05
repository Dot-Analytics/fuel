# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.fuel_model import run_fuel_model

app = FastAPI(title="DTI Fuel Model API")


class FuelRunRequest(BaseModel):
    run_date: str
    scenario: str | None = "default"
    top_n: int | None = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_model(req: FuelRunRequest):
    try:
        result = run_fuel_model(
            run_date=req.run_date,
            scenario=req.scenario,
            top_n=req.top_n,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

