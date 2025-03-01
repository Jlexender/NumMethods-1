import numpy as np
from fastapi import FastAPI, HTTPException
app = FastAPI()

from solver import solve
from eqSystem import EqSystemRequest

@app.post("/")
def main_endpoint(system: EqSystemRequest):
    try:
        return solve(system)
    except HTTPException as e:
        return {"error": e.detail}