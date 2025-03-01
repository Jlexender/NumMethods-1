from pydantic import BaseModel
from typing import List

class EqSystemRequest(BaseModel):
    matrix: List[List[float]]
    resultVector: List[float]
    accuracy: float

    