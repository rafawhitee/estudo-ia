import numpy as np
import matplotlib.pyplot as plt

# Vetores exemplo
v1 = np.array([2, 1])
v2 = np.array([4, 3])

# Fórmula da projeção ortogonal de v2 sobre v1
proj_v1 = (np.dot(v2, v1) / np.dot(v1, v1)) * v1
print(f"Projeção ortogonal de v2 sobre v1: {proj_v1}")

# Fórmula da projeção ortogonal de v1 sobre v2
proj_v2 = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
print(f"Projeção ortogonal de v1 sobre v2: {proj_v2}")

# Visualização
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v2')
plt.quiver(0, 0, proj_v1[0], proj_v1[1], angles='xy', scale_units='xy', scale=1, color='green', label='Projeção V2 sobre V1')
plt.quiver(0, 0, proj_v2[0], proj_v2[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Projeção V1 sobre V2')
plt.xlim(-1, 5)
plt.ylim(-1, 4)
plt.grid()
plt.legend()
plt.title("Projeção ortogonal de v1 sobre v2")
plt.show()