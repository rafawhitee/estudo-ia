import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

source = pd.read_csv('projects.csv')
source['finished'] = source.unfinished.map({0: 1, 1: 0})
sns.scatterplot(x="expected_hours", y="price", hue="finished", data=source)

x = source[['expected_hours', 'price']]
y = source['finished']

SEED = 20
train_x, test_x, train_y, test_y = train_test_split(
    x, y, stratify=y, random_state=SEED, test_size=0.25)

model = LinearSVC(dual='auto')
model.fit(train_x, train_y)
predictions = model.predict(test_x)

accuracy = accuracy_score(test_y, predictions)
print(f"accuracy --> {accuracy * 100}%")

print(f"train_y.value_counts --> {train_y.value_counts()}%")
print(f"test_y.value_counts --> {test_y.value_counts()}%")

base_line = np.ones(540)
accuracy_base_line = accuracy_score(test_y, base_line)
print(f"accuracy_base_line --> {accuracy_base_line * 100}%")

# A Acurácia foi baixa, então para melhorar a previsão, devemos nos basear da curva que o gráfico faz
# Se estiver a acima da curva o projeto tem que ter previsão de FINALIZADO
# Abaixo da curva, previsão de NÃO SER FINALIZADO

# vamos definir o valor mínimo/máximo do Eixo X e do Eixo Y
min_x = test_x.expected_hours.min()
max_x = test_x.expected_hours.max()

min_y = test_x.price.min()
max_y = test_x.price.max()

# a partir do começo e fim (de cada eixo: X e Y), será dividido por 100 pixels (variável abaixo)
pixels = 100

# cria os arrays com as posições
# começando no valor mínimo e vai somando 0.99 até o valor máximo de cada eixo
x_arange = np.arange(min_x, max_x, (max_x - min_x) / pixels)
y_arange = np.arange(min_y, max_y, (max_y - min_y) / pixels)

# cria os paredes ordenados
xx, yy = np.meshgrid(x_arange, y_arange)
points = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(points)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(test_x.expected_hours, test_x.price, c=test_y, s=1)
plt.show()