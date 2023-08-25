import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features (1 --> sim e 0 --> não)
# pelo longo?
# perna curta?
# late?

# cachorros
train_x  = []
train_x.append([0, 1, 1])
train_x.append([0, 1, 1])
train_x.append([1, 1, 1])
train_x.append([0, 0, 1])
train_x.append([1, 1, 1])
train_x.append([1, 1, 1])
train_x.append([1, 1, 1])
train_x.append([0, 0, 1])
train_x.append([1, 1, 1])
train_x.append([1, 1, 1])
train_x.append([0, 0, 1])

# porcos
train_x.append([0, 1, 0])
train_x.append([1, 0, 0])
train_x.append([1, 1, 0])
train_x.append([0, 0, 1])
train_x.append([0, 0, 0])
train_x.append([0, 1, 0])
train_x.append([0, 0, 0])
train_x.append([0, 1, 1])
train_x.append([1, 0, 0])
train_x.append([0, 1, 0])
train_x.append([1, 1, 1])


# 0 --> cachorro
# 1 --> porco
qt_dogs = 11
qt_pigs = 11
train_y = np.zeros(qt_dogs).tolist() + np.ones(qt_pigs).tolist()  # labels

# inicia o estimador do sklearn (para aprender sobre os dados)
model = LinearSVC(dual='auto')

# treina os dados
model.fit(train_x, train_y)

# cria um animal misterioso com as características
unique_test_x = [1, 1, 1]

# pergunta ao modelo com a estimativa (se é cachorro ou porco)
print(f"Predict Único --> {model.predict([unique_test_x])[0]}")

# teste x --> o que será validado para verificar se está correto
test_x = []
test_x.append([1, 0, 0])
test_x.append([1, 1, 1])
test_x.append([1, 0, 1])
test_x.append([1, 1, 0])
test_x.append([0, 1, 0])
test_x.append([0, 0, 0])
test_x.append([0, 1, 0])
test_x.append([0, 1, 1])
test_x.append([0, 1, 1])
test_x.append([1, 1, 1])
test_x.append([1, 1, 0])
test_x.append([0, 1, 1])

# teste y --> o resultado esperado dos animais misteriosos acima, para ver se o predict será correto
# 0 --> cachorro
# 1 --> porco
test_y = [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1]

# faz o predict dos misteriosos
predictions = model.predict(test_x)

print(f"Predict List --> {predictions}")

# calculando a taxa de acerto (se o predict dos misteriosos é igual ao teste_y)
accuracy = accuracy_score(test_y, predictions)
print(f"Taxa de Acerto --> {accuracy * 100}%")
