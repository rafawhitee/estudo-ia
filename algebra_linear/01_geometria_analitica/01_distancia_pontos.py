import math

# Exemplo 1: distância entre dois pontos no plano cartesiano
# Pontos A (x1, y1) e B (x2, y2)
x1, y1 = 1, 2
x2, y2 = 4, 6

# Fórmula da distância: sqrt((x2 - x1)**2 + (y2 - y1)**2)
distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
print(f"Distância entre A({x1},{y1}) e B({x2},{y2}): {distancia:.2f}")

# Exemplo 2: usando tuplas e função
def distancia_pontos(p1, p2):
    # p1 e p2 são tuplas (x, y)
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

A = (0, 0)
B = (3, 4)
print(f"Distância entre A{A} e B{B}: {distancia_pontos(A, B):.2f}")

# Exemplo 3: distância entre pontos no espaço 3D
x1, y1, z1 = 1, 2, 3
x2, y2, z2 = 4, 6, 9
distancia_3d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
print(f"Distância 3D entre ({x1},{y1},{z1}) e ({x2},{y2},{z2}): {distancia_3d:.2f}")