import numpy as np
from eqSystem import EqSystemRequest
from fastapi import HTTPException, status

MAX_ITERATIONS = 10000

def validate(matrix: EqSystemRequest):
    if len(matrix.matrix) == 0:
        return {"detail": "Пустая матрица"}
    if len(matrix.matrix) != len(matrix.matrix[0]):
        return {"detail": "Неквадратная матрица"}
    if any(len(row) != len(matrix.matrix) for row in matrix.matrix):
        return {"detail": "Некорректная матрица"}
    if len(matrix.resultVector) != len(matrix.matrix):
        return {"detail": "Некорректный вектор b"}
    return None

def solve(matrix: EqSystemRequest):
    if error := validate(matrix):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error["detail"])
    
    n = len(matrix.matrix)
    # simple iteration

    # check diagonal dominance
    for i in range(n):
        if 2 * abs(matrix.matrix[i][i]) <= sum(abs(matrix.matrix[i][j]) for j in range(n) if j != i):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Матрица не является диагонально доминируемой")

    x = np.zeros(n)
    x_new = np.zeros(n)
    accuracy = matrix.accuracy
    total_iterations = 0
    while total_iterations < MAX_ITERATIONS:
        total_iterations += 1
        for i in range(n):
            x_new[i] = matrix.resultVector[i]
            for j in range(n):
                if i != j:
                    x_new[i] -= matrix.matrix[i][j] * x[j]
            x_new[i] /= matrix.matrix[i][i]
        if np.linalg.norm(x_new - x) < accuracy:
            break
        x = x_new.copy()
    return x.tolist()
