
# Exemplos de posição relativa entre retas
# Retas paralelas: coeficientes angulares iguais
# Retas perpendiculares: produto dos coeficientes angulares igual a -1

def verifica_posicao_retas(m1, m2):
	if m1 == m2:
		print(f"Retas paralelas: m1 = {m1}, m2 = {m2}")
	elif m1 * m2 == -1:
		print(f"Retas perpendiculares: m1 = {m1}, m2 = {m2}")
	else:
		print(f"Retas concorrentes: m1 = {m1}, m2 = {m2}")

# Exemplos práticos
# Retas paralelas
m1 = 2
m2 = 2
verifica_posicao_retas(m1, m2)

# Retas perpendiculares
m3 = 0.5
m4 = -2
verifica_posicao_retas(m3, m4)

# Retas concorrentes (não paralelas nem perpendiculares)
m5 = 1
m6 = 3
verifica_posicao_retas(m5, m6)
