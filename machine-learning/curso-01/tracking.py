import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

source = pd.read_csv('tracking.csv')

x = source[["home", "how_it_works", "contact"]]

y = source["bought"]

# sempre separe bem os dados em 2 partes: para treinar e uma parte para testar (após treinar)
# no tracking.csv tem 99 linhas e 4 colunas, então vamos usar 75 para treinar e os outros 24 para testar

# pega os 75 primeiros para treinar o algoritmo
# train_x = x[:75]
# train_y = y[:75]

# pega a partir do 75 em diante (os 24 últimos no caso), para testar
# test_x = x[75:]
# test_y = y[75:]

# método nativo do sklearn para separar os dados em Treinos e Testes
# o SEED é para ele não fazer o random dos indexes e resultar em vários resultados diferentes
# o stratify é para manter a mesma proporção dos resultados (Y) de acordo com o Treino
SEED = 20
train_x, test_x, train_y, test_y = train_test_split(
    x, y, stratify=y, random_state=SEED, test_size=0.25)

print(
    f"Treinaremos com {len(train_x)} elementos e testaremos com {len(test_x)} elementos.")

model = LinearSVC(dual='auto')
model.fit(train_x, train_y)
predictions = model.predict(test_x)
print(f"predictions --> {predictions}")

accuracy = accuracy_score(test_y, predictions)
print(f"accuracy --> {accuracy * 100}%")

print(f"train_y.value_counts --> {train_y.value_counts()}%")
print(f"test_y.value_counts --> {test_y.value_counts()}%")
