import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

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
    X_normalizado, Y, test_size=0.3, random_state=123)

# model
model = BernoulliNB(binarize=0.44)

# treina
model.fit(X_treino, Y_treino)

# testa
predict_BNb = model.predict(X_teste)

print(predict_BNb)