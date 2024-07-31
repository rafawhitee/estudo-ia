import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

base_risco_credito = pd.read_csv(os.getcwd() + '\\database\\risco_credito.csv')

X_risco_credito = base_risco_credito.iloc[:, 0:4].values
Y_risco_credito = base_risco_credito.iloc[:, 4].values

# Fazendo o Pré-Processamento (convertendo os textos para numéricos)
X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

# Instancia o modelo Guaussian do Naive Bayes
naive_risco_credito = GaussianNB()
 
# Treina os dados para gerar a tabela de probabilidade
naive_risco_credito.fit(X_risco_credito, Y_risco_credito)

# Faz uma previsão, após treinar os dados e gerar a tabela de probabilidade
# 1) história boa, dívida alta, nenhuma garantia, renda maior que 35 mil
previsao = naive_risco_credito.predict([[0, 0, 1, 2]]) # os valoresnuméricos são os representantes das categorias pelo LabelEncoder
print(f"previsao --> {previsao}")

# Faz uma segunda previsão
# 2) história ruim, dívida alta, garantias adequada, renda menor que 15 mil
previsao_2 = naive_risco_credito.predict([[2, 0, 0, 0]]) # os valores numéricos são os representantes das categorias pelo LabelEncoder
print(f"previsao 2 --> {previsao_2}")
