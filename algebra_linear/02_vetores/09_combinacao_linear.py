import numpy as np

# Vetores base
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Vetor desejado para combinação linear
v = np.array([3, 2])

# Monta matriz com v1 e v2 como colunas
A = np.column_stack((v1, v2))
# Resolve a equação: a*v1 + b*v2 = v
sol = np.linalg.lstsq(A, v, rcond=None)[0]
print(f"Para v = {v}, os coeficientes são: a={sol[0]:.2f}, b={sol[1]:.2f}")

# Outro exemplo
v_ex2 = np.array([5, -1])
sol2 = np.linalg.lstsq(A, v_ex2, rcond=None)[0]
print(f"Para v = {v_ex2}, os coeficientes são: a={sol2[0]:.2f}, b={sol2[1]:.2f}")
