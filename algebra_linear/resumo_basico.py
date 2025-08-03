# Álgebra Linear com NumPy

import numpy as np

# 1. Vetores
# Vetores representam grandezas com direção e sentido, como velocidade ou força.
v1 = np.array([2, 3])
v2 = np.array([1, 4])
print("Soma de vetores:", v1 + v2)  # Exemplo: somar deslocamentos
print("Produto escalar:", np.dot(v1, v2))  # Exemplo: calcular trabalho (força x deslocamento)

# 2. Matrizes
# Matrizes podem representar sistemas de equações, imagens, ou transformações em gráficos.
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print("Soma de matrizes:\n", A + B)  # Exemplo: combinar dados de sensores
print("Multiplicação de matrizes:\n", np.dot(A, B))  # Exemplo: aplicar transformação em imagens

# 3. Transposta
# Transpor uma matriz é inverter linhas e colunas. Útil para manipular dados ou mudar perspectivas.
print("Transposta de A:\n", A.T)  # Exemplo: converter dados de formato linha para coluna

# 4. Determinante
# O determinante indica se uma matriz é invertível e mede o "volume" transformado por ela.
# Exemplo no mundo real: Em gráficos, se o determinante é zero, não há solução única para o sistema (ex: pontos colineares).
print("Determinante de A:", np.linalg.det(A))

# 5. Inversa
# A inversa de uma matriz é usada para "desfazer" uma transformação.
# Exemplo: Recuperar uma imagem original após uma transformação linear.
print("Inversa de A:\n", np.linalg.inv(A))

# 6. Sistema linear Ax = b
# Resolver sistemas lineares é essencial em engenharia, economia, física, etc.
# Exemplo: Encontrar valores de variáveis que satisfazem múltiplas restrições (ex: mistura de ingredientes).
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print("Solução do sistema Ax = b:", x)  # x = [1, 2]

# 7. Autovalores e autovetores
# Autovalores e autovetores mostram direções principais de transformação.
# Exemplo: Em Machine Learning, são usados em PCA para reduzir dimensões de dados.
eigvals, eigvecs = np.linalg.eig(A)
print("Autovalores de A:", eigvals)
print("Autovetores de A:\n", eigvecs)