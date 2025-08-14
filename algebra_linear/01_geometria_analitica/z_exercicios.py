import math
import numpy as np
from sympy import symbols, Matrix, factor, Eq, solve

###
### MÉTODOS UTILS
###
def compute_distance(A, B):
    return math.sqrt((A[1] - A[0])**2 + (B[1] - B[0])**2)

def compute_det(A, B, C):
    return np.linalg.det(np.array([
		[A[0], A[1], 1],
		[B[0], B[1], 1],
		[C[0], C[1], 1]
	]))
    
###
### 1. Demonstre que o triângulo de vértices A(8 , 2), B(3 , 7) e C(2 , 1) é isósceles. Em seguida, calcule seu perímetro.
###
print(f"\n Exercício 1: Demonstre que o triângulo de vértices A(8 , 2), B(3 , 7) e C(2 , 1) é isósceles. Em seguida, calcule seu perímetro.")
ex_1_A = (8, 2)
ex_1_B = (3, 7)
ex_1_C = (2, 1)

# Verifica se é isósceles
ex_1_AB = compute_distance(ex_1_A, ex_1_B)
ex_1_AC = compute_distance(ex_1_A, ex_1_C)
ex_1_BC = compute_distance(ex_1_B, ex_1_C)
print(f"\t Distâncias dos lados do triângulo:")
print(f"\t Lado AB: {ex_1_AB:.2f}")
print(f"\t Lado AC: {ex_1_AC:.2f}")
print(f"\t Lado BC: {ex_1_BC:.2f}")

is_isosceles = ex_1_AB == ex_1_AC or ex_1_AB == ex_1_BC or ex_1_AC == ex_1_BC
print(f"\t É isósceles: {is_isosceles}")

# Cálculo do perímetro
perimetro = ex_1_AB + ex_1_AC + ex_1_BC
print(f"\t Perímetro do triângulo: {perimetro:.2f}")
print(f"Fim do Exercício 1. \n")

###
### 2. Quais são os possíveis valores de c para que os pontos (c , 3), (2 , c) e (14, -3) sejam colineares?
###

print(f"\n Exercício 2: Quais são os possíveis valores de c para que os pontos (c , 3), (2 , c) e (14, -3) sejam colineares?")
c = symbols('c')
ex_2_A = (c, 3)
ex_2_B = (2, c)
ex_2_C = (14, -3)

# gera uma matriz com os pontos, incluíndo a variável C
ex_2_Matrix = Matrix([
    [ex_2_A[0],  ex_2_A[1], 1],
    [ex_2_B[0],  ex_2_B[1], 1],
    [ex_2_C[0],  ex_2_C[1], 1]
])
# calcula
print(f"\t matriz do exercício 2: {ex_2_Matrix}")

ex_2_Matrix_det = ex_2_Matrix.det()
print(f"\t Determinante: {ex_2_Matrix_det}")

sol = solve(Eq(ex_2_Matrix_det, 0), c)
print(f"\t Valores possíveis de c: {sol}")

print(f"Fim do Exercício 2. \n")