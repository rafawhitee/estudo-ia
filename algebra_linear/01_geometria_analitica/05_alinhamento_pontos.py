# Exemplo: alinhamento (colinearidade) de 3 pontos usando o determinante
# Se o determinante da matriz formada pelos pontos for zero, os pontos são colineares
# Pontos A(x1, y1), B(x2, y2), C(x3, y3)

import numpy as np

def pontos_colineares(A, B, C):
	# Monta a matriz dos pontos
	M = np.array([
		[A[0], A[1], 1],
		[B[0], B[1], 1],
		[C[0], C[1], 1]
	])
	det = np.linalg.det(M)
	print(f"Determinante: {det:.2f}")
	if abs(det) < 1e-10:
		print("Os pontos são colineares (alinhados).")
	else:
		print("Os pontos NÃO são colineares.")

# Exemplo prático:
A = (1, 2)
B = (2, 4)
C = (3, 6)
pontos_colineares(A, B, C)

# Exemplo de pontos não colineares
D = (0, 0)
E = (1, 2)
F = (2, 1)
pontos_colineares(D, E, F)
