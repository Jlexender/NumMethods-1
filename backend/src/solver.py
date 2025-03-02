import numpy as np
from eqSystem import EqSystemRequest
from fastapi import HTTPException, status

MAX_ITERATIONS = 10000
    
def set_max_iterations(iterations: int):
    global MAX_ITERATIONS
    MAX_ITERATIONS = iterations
    return {"message": f"MAX_ITERATIONS changed to {iterations}."}

def validate(A, b):
    if A.shape[0] != A.shape[1]:
        raise HTTPException(status_code=400, detail="Matrix A must be square.")
    if A.shape[0] != b.shape[0]:
        raise HTTPException(status_code=400, detail="Matrix A and vector b must have compatible dimensions.")
    if np.isnan(A).any() or np.isnan(b).any():
        raise HTTPException(status_code=400, detail="Matrix A and vector b must not contain NaNs.")
    if np.linalg.det(A) == 0:
        raise HTTPException(status_code=400, detail="Matrix A must be invertible.")

def ensure_diagonal_dominance(A, b):
    n = A.shape[0]
    dig_dom = True
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i][i])
        if np.abs(A[i][i]) < row_sum:
            dig_dom = False
            for j in range(i + 1, n):
                if np.abs(A[j][i]) >= np.sum(np.abs(A[j])) - np.abs(A[j][i]):
                    A[[i, j], :] = A[[j, i], :]
                    b[[i, j]] = b[[j, i]]
                    dig_dom = True
                    break
            if not dig_dom:
                return A, b, dig_dom
    return A, b, dig_dom


def solve(eqSystem: EqSystemRequest):
    try:
        A = np.array(eqSystem.coefficientMatrix)
        b = np.array(eqSystem.resultVector)
        validate(A, b)
        n = A.shape[0]
        x = np.copy(b)
        eps = eqSystem.accuracy
        
        A, b, dig_dom = ensure_diagonal_dominance(A, b)
        
        i = 0
        while i < MAX_ITERATIONS:
            x_new = np.zeros(n)
            for j in range(n):
                x_new[j] = b[j]
                for k in range(n):
                    if j != k:
                        x_new[j] -= A[j][k] * x[k]
                if A[j][j] == 0:
                    raise HTTPException(status_code=400, detail="Division by zero encountered in the matrix.")
                x_new[j] /= A[j][j]
            if np.max(np.abs(x_new - x)) < eps:
                return {"solution": x_new.tolist(), "iterations": i, 
                "diagonalDominance": dig_dom, "mat_std_norm": np.linalg.norm(A),
                "невязка": (np.dot(A, x_new) - b).tolist(),
                "step_error": np.abs(x_new - x).tolist()
                }

            x = x_new
            i += 1

        raise HTTPException(status_code=400, detail="Method did not converge.") 
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    


    
