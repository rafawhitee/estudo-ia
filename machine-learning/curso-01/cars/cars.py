import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

source = pd.read_csv('cars.csv')

# novas colunas
source['model_age'] = datetime.today().year - source.model_year
source['kms_per_year'] = source.mileage_per_year * 1.60934

x = source[["price", "model_age", "kms_per_year"]]
y = source["sold"]

SEED = 5
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, stratify=y)

print("Treinaremos com %d elementos e testaremos com %d elementos" %
      (len(raw_train_x), len(raw_train_x)))

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = SVC()
model.fit(train_x, train_y)
svc_predictions = model.predict(test_x)

dummy_classifier = DummyClassifier(strategy='stratified')
dummy_classifier.fit(train_x, train_y)
dummy_accuracy = dummy_classifier.score(test_x, test_y)
print(f"A acurácia do DummyClassifier foi de {dummy_accuracy * 100}%")

linear_svc_accuracy = accuracy_score(test_y, svc_predictions)
print(f"A acurácia do SVC foi de {linear_svc_accuracy * 100}%")
