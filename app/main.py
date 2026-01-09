# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.fuel_model import run_fuel_model

app = FastAPI(title="DTI Fuel Model API")


class FuelRunRequest(BaseModel):
    run_date: str
    scenario: str | None = "default"
    top_n: int | None = 5
    mpg: float | None = 6.5
    tank_capacity: float | None = 200.0
    initial_fuel: float | None = 100.0
    safety_buffer: float | None = 10.0
    stop_fee: float | None = 100.0


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
            mpg=req.mpg,
            tank_capacity=req.tank_capacity,
            initial_fuel=req.initial_fuel,
            safety_buffer=req.safety_buffer,
            stop_fee=req.stop_fee,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
