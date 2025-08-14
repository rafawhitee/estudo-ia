# Exemplos de posição relativa entre retas
# Retas paralelas: coeficientes angulares iguais
# Retas perpendiculares: produto dos coeficientes angulares igual a -1

import math

def angulo_entre_retas(m1, m2):
    # Caso especial: retas perpendiculares (produto dos coeficientes = -1)
    if (1 + m1 * m2) == 0:
        print("Ângulo entre as retas: 90.00 graus (retas perpendiculares)")
        return
    # Calcula o ângulo em radianos
    tan_theta = abs((m2 - m1) / (1 + m1 * m2))
    theta_rad = math.atan(tan_theta)
    # Converte para graus
    theta_deg = math.degrees(theta_rad)
    print(f"Ângulo entre as retas: {theta_deg:.2f} graus")

# Exemplos didáticos para calcular o ângulo entre duas retas
m1 = 1
m2 = -1
angulo_entre_retas(m1, m2)  # Deve dar 90 graus (perpendiculares)

m3 = 2
m4 = 2
angulo_entre_retas(m3, m4)  # Deve dar 0 graus (paralelas)

m5 = 0.5
m6 = 3
angulo_entre_retas(m5, m6)  # Ângulo qualquer
