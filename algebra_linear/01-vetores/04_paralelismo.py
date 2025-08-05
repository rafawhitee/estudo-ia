import numpy as np
import matplotlib.pyplot as plt

def sao_paralelos(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0] == 0

v1 = np.array([-3, 2])
v2 = np.array([6, -4])

v3 = np.array([6, 2])
v4 = np.array([12, 4])

print(f"v1 e v2 são paralelos? {sao_paralelos(v1, v2)}")
print(f"v3 e v4 são paralelos? {sao_paralelos(v3, v4)}")

# Vetores que vão se encontrar (colidir) no ponto (2, 3)
v1_exercicio = np.array([2, 3])      # parte da origem [0, 0]
v2_exercicio = np.array([-2, 3])     # parte de [4, 0]

print(f"v1_exercicio e v2_exercicio são paralelos? {sao_paralelos(v1_exercicio, v2_exercicio)}")

origem1 = [0, 0]
origem2 = [4, 0]

plt.quiver(*origem1, v1_exercicio[0], v1_exercicio[1], angles='xy', scale_units='xy', scale=1, color='red', label='v1_exercicio')
plt.quiver(*origem2, v2_exercicio[0], v2_exercicio[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v2_exercicio')
plt.scatter(2, 3, color='purple', label='Ponto de colisão (2,3)', zorder=5)
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.grid()
plt.legend()
plt.title("Exemplo: Vetores que colidem no ponto (2,3)")
plt.show()