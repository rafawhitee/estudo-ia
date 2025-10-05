import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# A normalização dos valores antes de treinar um modelo de ML é crucial para garantir que todas as features contribuam de maneira equilibrada para o aprendizado.
# Sem normalização, features com escalas maiores podem dominar o cálculo dos gradientes, levando a uma convergência lenta ou instável.
# Nos casos abaixo, a febre pode ser 37 a 42, já os outros são escalas até 10 ou 20, por isso a febre domina o cálculo, e por isso também, deve-se normalizar.

## Função para treinar o modelo SEM normalização
def treinar_sem_normalizacao():
    """Treina modelo SEM normalização"""
    class LogisticRegressionFromScratch:
        def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
            self.learning_rate = learning_rate
            self.max_iterations = max_iterations
            self.weights = None
            self.bias = None
            self.cost_history = []
        
        def sigmoid(self, z: np.ndarray) -> np.ndarray:
            z = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z))
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for i in range(self.max_iterations):
                z = np.dot(X, self.weights) + self.bias
                predictions = self.sigmoid(z)
                
                # Previne log(0)
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
                self.cost_history.append(cost)
                
                dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
                db = (1 / n_samples) * np.sum(predictions - y)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            return (predictions >= 0.5).astype(int)
    
    # Dados sintéticos simples
    X = np.array([
        [37.0, 3, 2, 4, 5],   # Resfriado
        [37.5, 4, 3, 5, 6],   # Resfriado
        [39.0, 8, 8, 9, 10],  # Gripe
        [40.0, 9, 9, 10, 12], # Gripe
    ])
    y = np.array([0, 0, 1, 1])
    
    print("🚫 TREINANDO SEM NORMALIZAÇÃO:")
    print(f"Dados originais:\n{X}")
    print(f"Escalas: Febre:{X[:,0].min():.1f}-{X[:,0].max():.1f}, Dor:{X[:,1].min()}-{X[:,1].max()}")
    
    model = LogisticRegressionFromScratch(learning_rate=0.001, max_iterations=1000)  # LR baixo!
    model.fit(X, y)
    
    print(f"Pesos finais: {model.weights}")
    print(f"Custo final: {model.cost_history[-1]:.6f}")
    print(f"Convergiu? {'Sim' if model.cost_history[-1] < 0.1 else 'NÃO'}")
    
    return model.cost_history, model.weights

def treinar_com_normalizacao():
    """Treina modelo COM normalização"""
    class LogisticRegressionFromScratch:
        def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
            self.learning_rate = learning_rate
            self.max_iterations = max_iterations
            self.weights = None
            self.bias = None
            self.cost_history = []
        
        def sigmoid(self, z: np.ndarray) -> np.ndarray:
            z = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z))
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for i in range(self.max_iterations):
                z = np.dot(X, self.weights) + self.bias
                predictions = self.sigmoid(z)
                
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
                self.cost_history.append(cost)
                
                dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
                db = (1 / n_samples) * np.sum(predictions - y)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            return (predictions >= 0.5).astype(int)
    
    # Mesmos dados, mas normalizados
    X = np.array([
        [37.0, 3, 2, 4, 5],
        [37.5, 4, 3, 5, 6],
        [39.0, 8, 8, 9, 10],
        [40.0, 9, 9, 10, 12],
    ])
    
    # Normalização Z-score
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    
    y = np.array([0, 0, 1, 1])
    
    print("\n✅ TREINANDO COM NORMALIZAÇÃO:")
    print(f"Dados normalizados:\n{X_norm}")
    print(f"Todas as features agora têm média≈0 e std≈1")
    
    model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)  # LR normal!
    model.fit(X_norm, y)
    
    print(f"Pesos finais: {model.weights}")
    print(f"Custo final: {model.cost_history[-1]:.6f}")
    print(f"Convergiu? {'Sim' if model.cost_history[-1] < 0.1 else 'NÃO'}")
    
    return model.cost_history, model.weights

if __name__ == "__main__":
    print("🧪 EXPERIMENTO: Normalização vs Sem Normalização")
    print("=" * 60)
    
    # Testa sem normalização
    cost_sem_norm, weights_sem_norm = treinar_sem_normalizacao()
    
    # Testa com normalização  
    cost_com_norm, weights_com_norm = treinar_com_normalizacao()
    
    # Visualiza diferenças
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(cost_sem_norm, label='Sem Normalização', color='red', linewidth=2)
    plt.plot(cost_com_norm, label='Com Normalização', color='green', linewidth=2)
    plt.title('Convergência da Função de Custo')
    plt.xlabel('Iteração')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Escala log para ver melhor
    
    plt.subplot(1, 3, 2)
    features = ['Febre', 'Dor Cabeça', 'Dor Corpo', 'Fadiga', 'Duração']
    x_pos = np.arange(len(features))
    plt.bar(x_pos - 0.2, weights_sem_norm, 0.4, label='Sem Normalização', color='red', alpha=0.7)
    plt.bar(x_pos + 0.2, weights_com_norm, 0.4, label='Com Normalização', color='green', alpha=0.7)
    plt.title('Pesos Aprendidos')
    plt.xlabel('Features')
    plt.ylabel('Peso')
    plt.xticks(x_pos, features, rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Dados exemplo para mostrar escalas
    dados_exemplo = np.array([[38.5, 6, 4, 7, 8]])
    dados_norm = (dados_exemplo - np.mean(dados_exemplo)) / np.std(dados_exemplo)
    
    plt.bar(['Original'], [np.max(dados_exemplo) - np.min(dados_exemplo)], color='red', alpha=0.7)
    plt.bar(['Normalizado'], [np.max(dados_norm) - np.min(dados_norm)], color='green', alpha=0.7)
    plt.title('Range dos Dados')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 RESUMO:")
    print(f"Sem normalização - Custo final: {cost_sem_norm[-1]:.6f}")
    print(f"Com normalização - Custo final: {cost_com_norm[-1]:.6f}")
    print(f"Melhoria: {((cost_sem_norm[-1] - cost_com_norm[-1]) / cost_sem_norm[-1] * 100):.1f}%")
    
    print(f"\n🎯 CONCLUSÃO:")
    print(f"• Com normalização: algoritmo converge MUITO mais rápido")
    print(f"• Pesos ficam mais balanceados")
    print(f"• Podemos usar learning rate maior")
    print(f"• Resultado mais estável e confiável")