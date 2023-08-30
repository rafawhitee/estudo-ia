import numpy as np

# classe que representa o Perceptron

# Fluxo
#  1) É executado a função do campo induzido
#  2) com o retorno do campo induzido é passado para a função de ativação (bipolar por exemplo) e ali ai ver se irá ativar o neurônio ou não
class Perceptron():
    def __init__(self, total_features):
        # Inicialização aleatória dos pesos e bias
        self.weights = np.random.rand(total_features)
        self.bias = np.random.rand()

    # função de ativação (bipolar)
    #   neurônio disparar --> retorna 1
    #   neurônio não dispara --> retorna 0
    def bipolar_activation(self, x):
        return 1 if x >= 0 else 0

    # para treinar o Perceptron
    def train(self, training_data, epochs, learning_rate):
        for _ in range(epochs):
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction

                # Atualiza os pesos e bias
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

    # para realizar previsões de novas entradas
    def predict(self, inputs):
        # Calcula o valor de ativação (campo induzido)
        activation = np.dot(self.weights, inputs) + self.bias
        return self.bipolar_activation(activation)


# total de features (eixo x)
#  se modificar aqui, tem que editar também no training_data e new_data
total_features = 2

# Dados de treinamento, que são as features (eixo x) --> peso e textura
training_data = [
    (np.array([0.3, 0.7]), 1),  # peso 0.3 e textura 0.7 --> Laranja (Valor 1)
    (np.array([0.6, 0.3]), 0),  # peso 0.6 e textura 0.3 --> Maçã (Valor 0)
    (np.array([0.2, 0.8]), 1),  # peso 0.2 e textura 0.8 --> Laranja (Valor 1)
    (np.array([0.8, 0.5]), 0),  # peso 0.8 e textura 0.5 --> Maçã (Valor 0)
]

# Teste com novos dados (quero descobrir a previsão de acordo com os testes)
new_data = [
    np.array([0.4, 0.6]),
    np.array([0.7, 0.4]),
    np.array([0.3, 0.7]),
]

# Criação e Treinamento do perceptron
perceptron = Perceptron(total_features=total_features)
perceptron.train(training_data, epochs=90000, learning_rate=0.11)


for data in new_data:
    prediction = perceptron.predict(data)
    if prediction == 1:
        print("Laranja")
    else:
        print("Maçã")
