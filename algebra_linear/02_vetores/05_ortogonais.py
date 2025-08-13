import numpy as np

def sao_ortogonais(v1, v2):
    return np.dot(v1, v2) == 0

# Ortogonalidade de vetores
# Dois vetores são ortogonais se o produto escalar entre eles é zero (ou seja, eles formam um ângulo de 90 graus)
v1 = np.array([2, 3])
v2 = np.array([1, 4])

v3 = np.array([1, 2])
v4 = np.array([-2, 1])

print(f"v1 e v2 são ortogonais? {sao_ortogonais(v1, v2)}")
print(f"v3 e v4 são ortogonais? {sao_ortogonais(v3, v4)}")