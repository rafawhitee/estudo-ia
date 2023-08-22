from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features (1 --> sim e 0 --> não)
# pelo longo?
# perna curta?
# late?

porco_1 = [0, 1, 0]
porco_2 = [0, 1, 1]
porco_3 = [1, 1, 0]

cachorro_1 = [0, 1, 1]
cachorro_2 = [1, 0, 1]
cachorro_3 = [1, 1, 1]

# cria uma lista com os animais acima
dados = [cachorro_1, cachorro_2, cachorro_3, porco_1, porco_2, porco_3]

# 0 --> cachorro
# 1 --> porco 
classes = [0, 0, 0, 1, 1, 1]

# inicia o estimador do sklearn (para aprender sobre os dados)
model = LinearSVC()

# manda ele aprender os dados
model.fit(dados, classes)

# cria um animal misterioso com as características
animal_misterioso = [1, 1, 1]

# pergunta ao modelo com a estimativa (se é cachorro ou porco)
print(f"Predict Único --> {model.predict([animal_misterioso])[0]}")

# pergunta ao modelos várias estimativas
animal_misterioso_2 = [1, 1, 1]
animal_misterioso_3 = [1, 0, 1]
animal_misterioso_4 = [1, 1, 0]
animal_misterioso_5 = [0, 1, 0]
animal_misterioso_6 = [0, 0, 0]
animal_misterioso_7 = [0, 1, 0]
animal_misterioso_8 = [0, 1, 1]
previsoes = model.predict([animal_misterioso_2, animal_misterioso_3, animal_misterioso_4, animal_misterioso_5, animal_misterioso_6, animal_misterioso_7, animal_misterioso_8])

# cria um array com as classes dos animais misteriosos
testes_classes = [0, 0, 1, 1, 1, 1, 0]

print(f"Predict mais de um --> {previsoes}")

# calculando a taxa de acerto
taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print(f"Taxa de Acerto --> {taxa_de_acerto * 100}%")