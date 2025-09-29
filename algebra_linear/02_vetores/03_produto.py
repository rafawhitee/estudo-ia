import numpy as np

v1 = np.array([2, 3])
v2 = np.array([3, 5])
v3 = np.array([1, 4])

# Produto escalar (dot product)
# Mede o quanto dois vetores apontam na mesma direção
produto_escalar_1 = np.dot(v1, v2)
print("\n Produto escalar de v1 e v2:", produto_escalar_1)

produto_escalar_2 = np.dot(v1, v3)
print("\n Produto escalar de v1 e v3:", produto_escalar_2)

# Multiplicação por escalar
escalar = 2
multiplicacao = escalar * v1
print("\n Multiplicação de v1 por escalar:", multiplicacao)