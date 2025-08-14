# Exemplo 1: ponto médio entre dois pontos no plano cartesiano
# Pontos A (x1, y1) e B (x2, y2)
x1, y1 = 1, 2
x2, y2 = 4, 6

# Fórmula do ponto médio: ((x1 + x2)/2, (y1 + y2)/2)
ponto_medio = ((x1 + x2)/2, (y1 + y2)/2)
print(f"Ponto médio entre A({x1},{y1}) e B({x2},{y2}): {ponto_medio}")

# Exemplo 2: usando tuplas e função
def ponto_medio_2d(p1, p2):
    # p1 e p2 são tuplas (x, y)
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

A = (0, 0)
B = (3, 4)
print(f"Ponto médio entre A{A} e B{B}: {ponto_medio_2d(A, B)}")

def ponto_medio_3d(p1, p2):
    # p1 e p2 são tuplas (x, y, z)
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2, (p1[2] + p2[2])/2)

# Exemplo 3: ponto médio entre pontos no espaço 3D
x1, y1, z1 = 1, 2, 3
x2, y2, z2 = 4, 6, 9
print(f"Ponto médio 3D entre ({x1},{y1},{z1}) e ({x2},{y2},{z2}): {ponto_medio_3d((x1, y1, z1), (x2, y2, z2))}")