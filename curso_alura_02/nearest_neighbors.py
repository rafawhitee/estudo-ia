import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

dados = pd.read_csv('clientes.csv')

# cliente aleatório (para prever se ela irá sair)
X_Maria = [[0, 0, 1, 1, 0, 0, 39.90, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]]

# modificação de forma manual
traducao_dic = {'Sim': 1,
                'Nao': 0}

dados_modificados = dados[['Conjuge', 'Dependentes',
                           'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)

# transformação pelo get_dummies
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'],
                                         axis=1))

# junção dos dados trasformados com os que já tinhamos
dados_final = pd.concat([dados_modificados, dummie_dados], axis=1)

# dividindo os dados em caracteristicas (X) e target (Y)
X = dados_final.drop('Churn', axis=1)
Y = dados_final['Churn']

# padronizar os dados
norm = StandardScaler()
X_normalizado = norm.fit_transform(X)
X_Maria_normalizado = norm.transform(pd.DataFrame(X_Maria, columns=X.columns))

# Distância Euclidiana de forma manual (entre a Maria e o primeiro cliente no X_normalizado)
distancia_maria_e_cliente_0_normalizado = np.sqrt(
    np.sum(np.square(X_Maria_normalizado - X_normalizado[0])))
print(
    f"distancia_maria_e_cliente_0_normalizado (manual): {distancia_maria_e_cliente_0_normalizado}")

#####
# Utilizando a lib para calcular e treinar já com a distância euclidiana e prever
#####

X_treino, X_teste, Y_treino, Y_teste = train_test_split(
    X_normalizado, Y, test_size=0.1, random_state=123)

knn = KNeighborsClassifier(metric='euclidean')

knn.fit(X_treino, Y_treino)

predict = knn.predict(X_teste)

print(f"confusion_matrix nearest-neighbors: {confusion_matrix(Y_teste, predict)}")
print(f"accuracy_score nearest-neighbors: {accuracy_score(Y_teste, predict)}")
print(f"precision_score nearest-neighbors: {precision_score(Y_teste, predict)}")
print(f"recall_score nearest-neighbors: {recall_score(Y_teste, predict)} \n\n")