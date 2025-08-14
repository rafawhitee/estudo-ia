import math

def compute_distance(A, B):
    return math.sqrt((A[1] - A[0])**2 + (B[1] - B[0])**2)
    
# 1. Demonstre que o triângulo de vértices A(8 , 2), B(3 , 7) e C(2 , 1) é isósceles. Em seguida, calcule seu perímetro.
print(f"\n Exercício 1: Demonstre que o triângulo de vértices A(8 , 2), B(3 , 7) e C(2 , 1) é isósceles. Em seguida, calcule seu perímetro.")
ex_1_A = (8, 2)
ex_1_B = (3, 7)
ex_1_C = (2, 1)

# Verifica se é isósceles
ex_1_AB = compute_distance(ex_1_A, ex_1_B)
ex_1_AC = compute_distance(ex_1_A, ex_1_C)
ex_1_BC = compute_distance(ex_1_B, ex_1_C)
print(f"Distâncias dos lados do triângulo:")
print(f"Lado AB: {ex_1_AB:.2f}")
print(f"Lado AC: {ex_1_AC:.2f}")
print(f"Lado BC: {ex_1_BC:.2f}")

is_isosceles = ex_1_AB == ex_1_AC or ex_1_AB == ex_1_BC or ex_1_AC == ex_1_BC
print(f"É isósceles: {is_isosceles}")

# Cálculo do perímetro
perimetro = ex_1_AB + ex_1_AC + ex_1_BC
print(f"Perímetro do triângulo: {perimetro:.2f}")
print(f"\n Fim do Exercício 1.")