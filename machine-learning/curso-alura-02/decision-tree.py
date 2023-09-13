import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# faz a separação entre Treino e Teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(
    X_normalizado, Y, test_size=0.1, random_state=123)

# model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# treina
model.fit(X_treino, Y_treino)

# testa
predict = model.predict(X_teste)

print(f"confusion_matrix decision-tree: {confusion_matrix(Y_teste, predict)}")
print(f"accuracy_score decision-tree: {accuracy_score(Y_teste, predict)}")
print(f"precision_score decision-tree: {precision_score(Y_teste, predict)}")
print(f"recall_score decision-tree: {recall_score(Y_teste, predict)} \n\n")