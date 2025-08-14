m3 = 2
m5 = 0.5
# Exemplo: distância entre um ponto e uma reta
# Reta na forma geral: Ax + By + C = 0
# Ponto P(x0, y0)
# Fórmula: d = |A*x0 + B*y0 + C| / sqrt(A**2 + B**2)

import math

def distancia_ponto_reta(A, B, C, x0, y0):
    numerador = abs(A*x0 + B*y0 + C)
    denominador = math.sqrt(A**2 + B**2)
    d = numerador / denominador
    print(f"Distância do ponto P({x0},{y0}) à reta {A}x + {B}y + {C} = 0: {d:.2f}")

# Exemplo prático:
# Reta: 2x - 3y + 4 = 0
# Ponto: (1, 2)
distancia_ponto_reta(2, -3, 4, 1, 2)

# Outro exemplo:
# Reta: x + y - 5 = 0
# Ponto: (3, 4)
distancia_ponto_reta(1, 1, -5, 3, 4)
