import numpy as np

# Vetores exemplo
v1 = np.array([2, 3])
v2 = np.array([1, 4])

# Produto escalar
dot_product = np.dot(v1, v2)

# Módulos (normas) dos vetores
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

# Cálculo do cosseno do ângulo
cos_theta = dot_product / (norm_v1 * norm_v2)

# Ângulo em radianos
theta_rad = np.arccos(cos_theta)
print(f"Ângulo em radianos entre v1 e v2: {theta_rad:.2f} radianos")

# Ângulo em graus
theta_deg = np.degrees(theta_rad)

print(f"Ângulo em graus entre v1 e v2: {theta_deg:.2f} graus")