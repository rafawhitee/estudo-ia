# Exemplo 1: equação da circunferência no plano cartesiano
# Centro (a, b) e raio r
a, b = 2, 3
r = 5

# Equação reduzida: (x - a)^2 + (y - b)^2 = r^2
print(f"Equação reduzida da circunferência: (x - {a})^2 + (y - {b})^2 = {r**2}")

# Equação geral: x^2 + y^2 + Dx + Ey + F = 0
# D = -2a, E = -2b, F = a^2 + b^2 - r^2
D = -2*a
E = -2*b
F = a**2 + b**2 - r**2
print(f"Equação geral da circunferência: x^2 + y^2 + ({D})x + ({E})y + ({F}) = 0")

# Exemplo 2: usando função para qualquer centro e raio
def equacao_circunferencia(centro, raio):
    a, b = centro
    r = raio
    D = -2*a
    E = -2*b
    F = a**2 + b**2 - r**2
    print(f"Centro: {centro}, Raio: {raio}")
    print(f"Equação reduzida: (x - {a})^2 + (y - {b})^2 = {r**2}")
    print(f"Equação geral: x^2 + y^2 + ({D})x + ({E})y + ({F}) = 0")

equacao_circunferencia((0, 0), 2)