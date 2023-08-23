from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features (1 --> sim e 0 --> não)
# pelo longo?
# perna curta?
# late?

# cachorros
animal_1 = [0, 1, 1]
animal_2 = [0, 1, 1]
animal_3 = [1, 1, 1]
animal_4 = [0, 0, 1]
animal_5 = [1, 1, 1]
animal_6 = [1, 1, 1]
animal_7 = [1, 1, 1]
animal_8 = [0, 0, 1]

# porcos
animal_9 = [0, 1, 0]
animal_10 = [1, 0, 0]
animal_11 = [1, 1, 0]
animal_12 = [0, 0, 0]

# cria uma lista com os animais acima
train_x = [animal_1, animal_2, animal_3, animal_4, animal_5, animal_6,
           animal_7, animal_8, animal_9, animal_10, animal_11, animal_12]

# 0 --> cachorro
# 1 --> porco
train_y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # labels

# inicia o estimador do sklearn (para aprender sobre os dados)
model = LinearSVC(dual='auto')

# treina os dados
model.fit(train_x, train_y)

# cria um animal misterioso com as características
animal_misterioso_0 = [1, 1, 1]

# pergunta ao modelo com a estimativa (se é cachorro ou porco)
print(f"Predict Único --> {model.predict([animal_misterioso_0])[0]}")

# cria alguns animais misteriosos (não sabe se é Porco ou Cachorro)
animal_misterioso_1 = [1, 0, 0]
animal_misterioso_2 = [1, 1, 1]
animal_misterioso_3 = [1, 0, 1]
animal_misterioso_4 = [1, 1, 0]
animal_misterioso_5 = [0, 1, 0]
animal_misterioso_6 = [0, 0, 0]
animal_misterioso_7 = [0, 1, 0]
animal_misterioso_8 = [0, 1, 1]

# teste x --> o que será validado para verificar se está correto
teste_x = [animal_misterioso_1, animal_misterioso_2, animal_misterioso_3, animal_misterioso_4,
           animal_misterioso_5, animal_misterioso_6, animal_misterioso_7, animal_misterioso_8]

# teste y --> o resultado esperado dos animais misteriosos acima, para ver se o predict será correto
testes_y = [1, 0, 0, 1, 1, 1, 1, 0]

# faz o predict dos misteriosos
previsoes = model.predict(teste_x)

print(f"Predict List --> {previsoes}")

# calculando a taxa de acerto (se o predict dos misteriosos é igual ao teste_y)
taxa_de_acerto = accuracy_score(testes_y, previsoes)
print(f"Taxa de Acerto --> {taxa_de_acerto * 100}%")
