import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

np.random.seed(42)

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Função sigmoid para converter valores em probabilidades (0-1)"""
        # Previne overflow numérico
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina o modelo com os dados fornecidos"""
        # Inicializa pesos e bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Calcula o custo (log loss)
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Backward pass (gradientes)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Atualiza parâmetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a função de custo (log loss)"""
        # Previne log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições (0 ou 1)"""
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

# Exemplo prático: Classificação Gripe vs Resfriado
def gerar_dados_exemplo() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Gera dados sintéticos para classificação gripe vs resfriado"""
    # Features: [febre_celsius, dor_cabeca_intensidade, dor_corpo_intensidade, fadiga_intensidade, duracao_dias]
    feature_names = ['Febre (°C)', 'Dor de Cabeça (1-10)', 'Dor no Corpo (1-10)', 'Fadiga (1-10)', 'Duração (dias)']
    
    # Dados do resfriado (classe 0) - sintomas mais leves
    n_resfriados = 150
    resfriado_febre = np.random.normal(37.2, 0.8, n_resfriados)  # Febre baixa: 37.2°C ± 0.8°C
    resfriado_dor_cabeca = np.random.normal(3, 1.5, n_resfriados)  # Dor leve: 3 ± 1.5
    resfriado_dor_corpo = np.random.normal(2, 1, n_resfriados)  # Dor corporal leve: 2 ± 1
    resfriado_fadiga = np.random.normal(4, 1.5, n_resfriados)  # Fadiga moderada: 4 ± 1.5
    resfriado_duracao = np.random.normal(5, 2, n_resfriados)  # Duração: 5 ± 2 dias
    
    # Dados da gripe (classe 1) - sintomas mais intensos
    n_gripes = 150
    gripe_febre = np.random.normal(39.5, 1, n_gripes)  # Febre alta: 39.5°C ± 1°C
    gripe_dor_cabeca = np.random.normal(8, 1.5, n_gripes)  # Dor intensa: 8 ± 1.5
    gripe_dor_corpo = np.random.normal(8.5, 1, n_gripes)  # Dor corporal intensa: 8.5 ± 1
    gripe_fadiga = np.random.normal(9, 1, n_gripes)  # Fadiga severa: 9 ± 1
    gripe_duracao = np.random.normal(10, 3, n_gripes)  # Duração: 10 ± 3 dias
    
    # Garante que os valores estejam em ranges realistas
    resfriado_febre = np.clip(resfriado_febre, 36, 38.5)
    resfriado_dor_cabeca = np.clip(resfriado_dor_cabeca, 1, 10)
    resfriado_dor_corpo = np.clip(resfriado_dor_corpo, 1, 10)
    resfriado_fadiga = np.clip(resfriado_fadiga, 1, 10)
    resfriado_duracao = np.clip(resfriado_duracao, 2, 14)
    
    gripe_febre = np.clip(gripe_febre, 37.5, 42)
    gripe_dor_cabeca = np.clip(gripe_dor_cabeca, 1, 10)
    gripe_dor_corpo = np.clip(gripe_dor_corpo, 1, 10)
    gripe_fadiga = np.clip(gripe_fadiga, 1, 10)
    gripe_duracao = np.clip(gripe_duracao, 3, 21)
    
    # Combina os dados
    X = np.vstack([
        np.column_stack([resfriado_febre, resfriado_dor_cabeca, resfriado_dor_corpo, resfriado_fadiga, resfriado_duracao]),
        np.column_stack([gripe_febre, gripe_dor_cabeca, gripe_dor_corpo, gripe_fadiga, gripe_duracao])
    ])
    
    y = np.hstack([np.zeros(n_resfriados), np.ones(n_gripes)])  # 0=resfriado, 1=gripe
    
    return X, y, feature_names

def normalizar_dados(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza os dados (importante para convergência)"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def dividir_dados(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide os dados em treino e teste"""
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Embaralha os índices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de avaliação"""
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }

# Exemplo de uso
if __name__ == "__main__":
    print("🤒 vs 🤧 Classificador Médico - Gripe vs Resfriado!")
    print("=" * 60)
    
    # 1. Gera dados
    X, y, feature_names = gerar_dados_exemplo()
    print(f"Dataset criado: {len(X)} amostras, {X.shape[1]} sintomas")
    print(f"Sintomas: {feature_names}")
    print(f"Classes: {np.unique(y)} (0=Resfriado, 1=Gripe)")
    
    # 2. Divide em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    print(f"Treino: {len(X_train)} casos")
    print(f"Teste: {len(X_test)} casos")
    
    # 3. Normaliza os dados
    X_train_norm, mean, std = normalizar_dados(X_train)
    X_test_norm = (X_test - mean) / std  # Usa mesma normalização do treino
    
    # 4. Treina o modelo
    print("\n🤖 Treinando diagnóstico automático...")
    model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train_norm, y_train)
    
    # 5. Faz predições
    y_pred_train = model.predict(X_train_norm)
    y_pred_test = model.predict(X_test_norm)
    
    # 6. Avalia resultados
    print("\n📊 Resultados do Diagnóstico:")
    train_metrics = calcular_metricas(y_train, y_pred_train)
    test_metrics = calcular_metricas(y_test, y_pred_test)
    
    print(f"Treino - Accuracy: {train_metrics['accuracy']:.3f}")
    print(f"Teste  - Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Teste  - Precision: {test_metrics['precision']:.3f}")
    print(f"Teste  - Recall: {test_metrics['recall']:.3f}")
    print(f"Teste  - F1-Score: {test_metrics['f1_score']:.3f}")
    
    # 7. Exemplo de diagnóstico individual
    print("\n🔮 Exemplo de diagnóstico:")
    # Paciente exemplo: febre=38.5°C, dor_cabeca=6, dor_corpo=4, fadiga=7, duracao=3
    paciente = np.array([[38.5, 6, 4, 7, 3]])
    paciente_norm = (paciente - mean) / std
    
    prob = model.predict_proba(paciente_norm)[0]
    pred = model.predict(paciente_norm)[0]
    
    diagnostico = "Gripe" if pred == 1 else "Resfriado"
    confianca = prob if pred == 1 else (1 - prob)
    
    print(f"Paciente com sintomas {paciente[0]} é diagnosticado como: {diagnostico}")
    print(f"Confiança do diagnóstico: {confianca:.1%}")
    
    # 8. Mostra importância dos sintomas
    print(f"\n💡 Importância dos sintomas (pesos aprendidos):")
    for i, (sintoma, peso) in enumerate(zip(feature_names, model.weights)):
        print(f"  {sintoma}: {peso:.3f}")
    print(f"  Bias: {model.bias:.3f}")
    
    # 9. Visualiza a função de custo e distribuição dos dados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(model.cost_history)
    plt.title('Função de Custo Durante o Treino')
    plt.xlabel('Iteração')
    plt.ylabel('Log Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='lightblue', label='Resfriado', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Gripe', alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Febre vs Dor de Cabeça')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X[y==0, 2], X[y==0, 3], c='lightblue', label='Resfriado', alpha=0.7)
    plt.scatter(X[y==1, 2], X[y==1, 3], c='red', label='Gripe', alpha=0.7)
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.title('Dor no Corpo vs Fadiga')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 10. Casos de teste interessantes
    print("\n🩺 Testando casos específicos:")
    casos_teste = [
        [36.8, 2, 1, 3, 4],    # Resfriado leve
        [39.8, 9, 9, 10, 12],  # Gripe severa
        [37.5, 5, 5, 6, 7],    # Caso limítrofe
    ]
    
    for i, caso in enumerate(casos_teste):
        caso_norm = (np.array([caso]) - mean) / std
        prob = model.predict_proba(caso_norm)[0]
        pred = model.predict(caso_norm)[0]
        diag = "Gripe" if pred == 1 else "Resfriado"
        conf = prob if pred == 1 else (1 - prob)
        
        print(f"Caso {i+1}: {caso} → {diag} (confiança: {conf:.1%})")