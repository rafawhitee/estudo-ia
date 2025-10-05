import os
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

path = os.getcwd() + '\\pkl\\credit.pkl'
with open(path, 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, Y_credit_treinamento)

previsoes = naive_credit_data.predict(X_credit_teste)
print(f"Previsões --> {previsoes}")

print(f"Accuracy --> {accuracy_score(Y_credit_teste, previsoes)}")
print(f"Confusion Matrix --> {confusion_matrix(Y_credit_teste, previsoes)}")