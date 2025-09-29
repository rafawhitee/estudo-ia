import numpy as np

# Imagine que você tem 2 coordenadas de pontos A e B
# A = (x1, y1) e B = (x2, y2
A = np.array([2, 3])
B = np.array([1, 4])

# Para calcular o vetor AB, subtraímos as coordenadas de A de B
# OBS: sempre a final menos o inicial
vetor_ab = B - A
print("Vetor AB:", vetor_ab)

# Para calcular o vetor BA, subtraímos as coordenadas de A de B
# OBS: sempre a final menos o inicial (agora a final é o A, então é A - B)
vetor_ba = A - B
print("Vetor BA:", vetor_ba)