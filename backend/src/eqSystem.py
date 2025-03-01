from pydantic import BaseModel
from typing import List

class EqSystemRequest(BaseModel):
    coefficientMatrix: List[List[float]]
    resultVector: List[float]
    accuracy: float

    