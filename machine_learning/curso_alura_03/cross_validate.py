import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

# inicia o SEED e altera no numpy
SEED = 3000
np.random.seed(SEED)

# url com os dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"

# carrega os dados da uri
dados = pd.read_csv(uri)

# remove a primeira coluna (index)
dados = dados.drop(columns=["Unnamed: 0"], axis=1)

# vamos usar o modelo de classificação "Árvore de Decisão"
modelo = DecisionTreeClassifier(max_depth=2) # coloca o máximo 2 de profundidade

# separa em X e Y
x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

# utilizando o cross_validate, você passa o modelo, os valores de X e Y e em quantas partes você deseja dividir os dados (parametro 'cv')
resultado = cross_validate(modelo, x, y, cv = 5)
media = resultado["test_score"].mean() # média
desvio_padrao = resultado["test_score"].std() # desvio padrão

valor_min = (media - 2 * desvio_padrao) * 100
valor_max = (media + 2  * desvio_padrao) * 100

print(f"Valor Min --> {valor_min} %")
print(f"Valor Max --> {valor_max} %")