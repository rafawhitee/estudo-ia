import numpy as np

v1 = np.array([1, 2])
v2 = np.array([4, 6])

print(f"Calculando distância euclidiana entre v1 e v2:")
dif = v1 - v2
print(f"v1 - v2 --> {dif}")

dif_squared = dif**2
print(f"(v1 - v2)**2 --> {dif_squared}")

sum_dif_squared = np.sum(dif_squared)
print(f"np.sum((v1 - v2)**2) --> {sum_dif_squared}")

dist_euclidiana = np.sqrt(sum_dif_squared)
print(f"dist_euclidiana --> {dist_euclidiana}")