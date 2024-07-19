# %%
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


nome_arquivo_modelo = 'random_forest_model.pkl'
model = None
try:
    model = joblib.load(nome_arquivo_modelo)
except:
    print("Não conseguiu carregar o PKL, criando o Modelo para treinar")
    model = RandomForestClassifier(random_state=42) # Inicializa o modelo RandomForestClassifier
    model.fit(X_train, y_train) # Treina o modelo
    joblib.dump(model, nome_arquivo_modelo) # Salva o modelo treinado no arquivo

# Previsões
y_pred = model.predict(X_test)

# faz um predict de teste (tem que retornar a classe 2)
print(model.predict([[5.9, 3.0,  5.1, 1.8]]))

# Avaliação do modelo
#print("Relatório de Classificação:")
#print(classification_report(y_test, y_pred, target_names=iris.target_names))
#print("Matriz de Confusão:")
#print(confusion_matrix(y_test, y_pred))
# %%
