import numpy as np

class KohonenNetwork:
    def __init__(self, input_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights = np.random.rand(output_dim, input_dim)

    def find_winner(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=1)
        return np.argmin(distances)

    def update_weights(self, input_vector, winner_index):
        delta_weights = self.learning_rate * (input_vector - self.weights)
        self.weights += delta_weights

    def train(self, input_data, epochs=100):
        for _ in range(epochs):
            for input_vector in input_data:
                winner_index = self.find_winner(input_vector)
                self.update_weights(input_vector, winner_index)

    def predict(self, input_vector):
        winner_index = self.find_winner(input_vector)
        return self.weights[winner_index]

# Exemplo de uso
if __name__ == "__main__":
    input_data = np.array([[0.1, 0.2],
                           [0.4, 0.35],
                           [0.8, 0.6],
                           [0.9, 0.8]])

    # Configurar a rede de Kohonen
    input_dim = 2
    output_dim = 2
    learning_rate = 0.1
    epochs = 100

    kohonen_net = KohonenNetwork(input_dim, output_dim, learning_rate)

    # Treinar a rede de Kohonen
    kohonen_net.train(input_data, epochs)

    # Prever o cluster para um novo vetor de entrada
    new_input = np.array([0.2, 0.3])
    predicted_cluster = kohonen_net.predict(new_input)
    print("Vetor de entrada:", new_input)
    print("Cluster previsto:", predicted_cluster)
