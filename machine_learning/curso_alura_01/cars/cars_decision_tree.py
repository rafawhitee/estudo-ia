import numpy as np
import pandas as pd
import graphviz
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, stratify=y)

print("Treinaremos com %d elementos e testaremos com %d elementos" %
      (len(train_x), len(test_x)))

model = DecisionTreeClassifier(max_depth=5)
model.fit(train_x, train_y)
predictions = model.predict(test_x)

dummy_classifier = DummyClassifier(strategy='stratified')
dummy_classifier.fit(train_x, train_y)
dummy_accuracy = dummy_classifier.score(test_x, test_y)

dot_data = export_graphviz(model, out_file=None, feature_names=x.columns, filled = True, rounded = True, class_names = ['não', 'sim'])
graphic = graphviz.Source(dot_data)
graphic.view()

print(f"A acurácia do DummyClassifier foi de {dummy_accuracy * 100}%")

decision_tree_accuracy = accuracy_score(test_y, predictions)
print(f"A acurácia do DecisionTreeClassifier foi de {decision_tree_accuracy * 100}%")
