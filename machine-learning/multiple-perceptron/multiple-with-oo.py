import numpy as np

# Utilizando Orientação a Objetos
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Inicialize os pesos das camadas ocultas e de saída aleatoriamente
        self.weights_input_hidden = np.random.rand(input_dim, hidden_dim)
        self.weights_hidden_output = np.random.rand(hidden_dim, output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        # Propagação para a camada oculta
        hidden_input = np.dot(input_data, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)

        # Propagação para a camada de saída
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        predicted_output = self.sigmoid(output_input)

        return predicted_output, hidden_output

    def backward(self, input_data, target, predicted_output, hidden_output):
        # Calcula o erro na camada de saída
        output_error = target - predicted_output
        output_delta = output_error * self.sigmoid_derivative(predicted_output)

        # Calcula o erro na camada oculta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        # Atualiza os pesos
        self.weights_hidden_output += hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += input_data.T.dot(hidden_delta) * self.learning_rate

    def train(self, input_data, target, epochs=10000):
        for epoch in range(epochs):
            predicted_output, hidden_output = self.forward(input_data)
            self.backward(input_data, target, predicted_output, hidden_output)

    def predict(self, input_data):
        predicted_output, _ = self.forward(input_data)
        return predicted_output

# Exemplo de uso
if __name__ == "__main__":
    # Dados de treinamento (peso e pH) e rótulos (1 para laranja, 0 para maçã)
    input_data = np.array([[150, 3.5],
                           [200, 4.0],
                           [100, 3.6],
                           [120, 3.2]])
    
    target = np.array([[1],
                       [1],
                       [0],
                       [0]])

    # Configurar a rede neural
    input_dim = 2
    hidden_dim = 4
    output_dim = 1
    learning_rate = 0.1
    epochs = 10000

    neural_net = NeuralNetwork(input_dim, hidden_dim, output_dim, learning_rate)

    # Treinar a rede neural
    neural_net.train(input_data, target, epochs)

    # Prever se uma fruta é laranja ou maçã
    new_data = np.array([[130, 3.7]])
    prediction = neural_net.predict(new_data)
    
    if prediction >= 0.5:
        print("É uma laranja!")
    else:
        print("É uma maçã!")
