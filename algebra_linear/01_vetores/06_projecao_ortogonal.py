import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import imageio

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

# Vetores: um bem maior que o outro
v_menor = np.array([1, 1])
v_maior = np.array([4, 3])

# Projeção ortogonal do maior sobre o menor
proj = (np.dot(v_maior, v_menor) / np.dot(v_menor, v_menor)) * v_menor

frames = []
steps = 20
for alpha in np.linspace(0, 1, steps):
    fig, ax = plt.subplots(figsize=(5, 5))
    # Vetor menor
    ax.quiver(0, 0, v_menor[0], v_menor[1], angles='xy', scale_units='xy', scale=1, color='red', label='Menor')
    # Vetor maior
    ax.quiver(0, 0, v_maior[0], v_maior[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Maior')
    # Projeção animada
    proj_anim = alpha * proj
    ax.quiver(0, 0, proj_anim[0], proj_anim[1], angles='xy', scale_units='xy', scale=1, color='green', label='Projeção')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.grid()
    ax.legend()
    ax.set_title("Projeção ortogonal do maior sobre o menor")
    # Salva frame
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(image)
    plt.close(fig)

# Salva o gif
imageio.mimsave('projecao_ortogonal.gif', frames, duration=0.08)
print("GIF salvo como 'projecao_ortogonal.gif'")