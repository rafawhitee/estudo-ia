import os
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

path = os.getcwd() + '\\pkl\\census.pkl'
with open(path, 'rb') as f:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

naive_census_data = GaussianNB()
naive_census_data.fit(X_census_treinamento, Y_census_treinamento)

previsoes = naive_census_data.predict(X_census_teste)
print(f"Previsões --> {previsoes}")

print(f"Accuracy --> {accuracy_score(Y_census_teste, previsoes)}")
print(f"Confusion Matrix --> {confusion_matrix(Y_census_teste, previsoes)}")