import numpy as np

# O que é um vetor?
# Vetor é uma sequência ordenada de números que pode representar posição, velocidade, força, etc.

#####
##### CRIANDO VETORES
#####
print("\n Criando vetores:")
v1 = np.array([2, 3])
v2 = np.array([3, 5])
v3 = np.array([1, 4])

print("\t v1:", v1)
print("\t v3:", v3)

#####
##### IGUALDADE DE VETORES
#####
# tem que ser mesmo tamanho e também os mesmos valores
print("\n Igualdade de vetores:")
print("\t v1 é igual a v2?", np.array_equal(v1, v2))  # True
print("\t v1 é igual a v3?", np.array_equal(v1, v3))  # False

# Norma do vetor --> distância da origem até o ponto representado pelo vetor
# Também chamado de módulo do vetor
modulo_v1 = np.linalg.norm(v1)
print("\n Módulo (norma) de v1:", modulo_v1)

#####
##### TIPOS DE VETORES  
#####
print("\n Tipos de vetores:")

# 1. Vetor nulo
nulo = np.array([0, 0])
print("\t 1. Vetor nulo:", nulo)

# 2. Vetor simétrico/oposto
vetor_01 = np.array([2, 3])
simetrico = -vetor_01
print(f"\t 2. Vetor 01: {vetor_01} e seu simétrico/oposto: {simetrico}")

# 3. Vetor unitário
# Vetor com módulo (norma) igual a 1
vetor_unitario = vetor_01 / np.linalg.norm(vetor_01)
print("\t 3. Vetor unitário de vetor_01", vetor_unitario)

# 4. Vetor colineares/paralelos
# Vetores que estão na mesma direção, mas podem ter sentidos diferentes
vetor_colinear = 2 * vetor_unitario
print("\t 4. Vetor colinear de vetor_01", vetor_colinear)

# 5. Vetor linha
linha = np.array([[1, 2, 3]])
print("\t 5. Vetor linha:", linha)

# 6. Vetor coluna
coluna = np.array([[1], [2], [3]])
print("\t 6. Vetor coluna:\n", coluna)

# Direção e sentido
# Vetores têm direção (orientação) e sentido (para onde apontam)
# Exemplo: v1 = [2, 3] aponta para o quadrante superior direito do plano cartesiano

# Resumo:
# Vetores são fundamentais para representar grandezas físicas e dados