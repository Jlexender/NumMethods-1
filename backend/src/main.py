import numpy as np
from fastapi import FastAPI, HTTPException
app = FastAPI()

from solver import solve, set_max_iterations
from eqSystem import EqSystemRequest

@app.post("/")
def main_endpoint(system: EqSystemRequest):
    return solve(system)

@app.put("/change")
def change_endpoint(iterations: int):
    return set_max_iterations(iterations)