# equação da reta que passa por dois pontos no plano cartesiano
# Pontos A (x1, y1) e B (x2, y2)
x1, y1 = 1, 2
x2, y2 = 4, 6

# coeficiente angular
m = (y2 - y1) / (x2 - x1)

# Equação reduzida: y = m*x + b
# Para encontrar b, use um dos pontos:
b = y1 - m*x1
print(f"Equação reduzida da reta: y = {m:.2f}x + {b:.2f}")

# Equação geral: Ax + By + C = 0
# A = y1 - y2, B = x2 - x1, C = x1*y2 - x2*y1
A_geral = y1 - y2
B_geral = x2 - x1
C_geral = x1*y2 - x2*y1
print(f"Equação geral da reta: {A_geral}x + {B_geral}y + {C_geral} = 0")

# Exemplo 2: usando função para qualquer par de pontos
def equacao_reta(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    A_geral = y1 - y2
    B_geral = x2 - x1
    C_geral = x1*y2 - x2*y1
    print(f"Pontos: {p1}, {p2}")
    print(f"Equação reduzida: y = {m:.2f}x + {b:.2f}")
    print(f"Equação geral: {A_geral}x + {B_geral}y + {C_geral} = 0")

equacao_reta((0, 0), (3, 4))