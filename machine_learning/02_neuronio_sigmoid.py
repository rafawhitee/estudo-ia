import numpy as np

class Neuronio:
    """
    Um neurônio artificial básico que simula o comportamento de um neurônio biológico.
    Recebe múltiplas entradas, aplica pesos, soma tudo e passa por uma função de ativação.
    """
    
    def __init__(self, n_entradas):
        """
        Inicializa o neurônio com pesos aleatórios.
        
        Exemplo: Neuronio(2) cria um neurônio que aceita 2 entradas
        - self.pesos = [-0.05, 0.12]  (valores aleatórios pequenos)
        - self.bias = 0.0             (ajuste independente)
        """
        # Pesos aleatórios pequenos + bias
        self.pesos = np.random.normal(0, 0.1, n_entradas)
        self.bias = 0.0
    
    def sigmoid(self, z):
        """
        Função de ativação sigmoid: converte qualquer número real em valor entre 0 e 1.
        
        Exemplos:
        - sigmoid(-5) = 0.007  (quase 0)
        - sigmoid(0)  = 0.5    (meio termo) 
        - sigmoid(5)  = 0.993  (quase 1)
        
        É como um "interruptor suave" que decide se o neurônio "dispara" ou não.
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, entradas):
        """
        Propagação para frente: calcula a saída do neurônio.
        
        Passos:
        1. Soma ponderada: z = entrada1*peso1 + entrada2*peso2 + bias
        2. Ativação: saída = sigmoid(z)
        
        Exemplo com entradas=[1, 0], pesos=[0.5, -0.3], bias=0.1:
        - z = 1*0.5 + 0*(-0.3) + 0.1 = 0.6
        - saída = sigmoid(0.6) = 0.646
        """
        # Converte para array NumPy se necessário
        entradas = np.array(entradas)
        # Soma ponderada + ativação
        z = np.dot(entradas, self.pesos) + self.bias
        return self.sigmoid(z)
    
    def treinar(self, entradas, target, lr=0.5):
        """
        Treina o neurônio ajustando pesos baseado no erro.
        
        Algoritmo (Gradient Descent):
        1. Faz predição
        2. Calcula erro = target - predição  
        3. Calcula quanto cada peso deve mudar
        4. Atualiza pesos na direção que reduz o erro
        
        Exemplo: target=1, predição=0.3
        - erro = 1 - 0.3 = 0.7 (erro positivo = neurônio deve "disparar mais")
        - aumenta pesos das entradas ativas
        - lr controla velocidade do aprendizado
        """
        # Converte para array NumPy
        entradas = np.array(entradas)
        
        # Predição
        predicao = self.forward(entradas)
        
        # Erro e gradientes
        erro = target - predicao
        derivada = predicao * (1 - predicao)  # derivada sigmoid
        
        # Atualiza pesos
        self.pesos += lr * erro * derivada * entradas
        self.bias += lr * erro * derivada
        
        return erro

# Exemplo: Neurônio aprendendo função AND
if __name__ == "__main__":
    """
    Demonstração: ensina um neurônio a simular porta lógica AND
    
    Função AND:
    - 0 AND 0 = 0 (falso E falso = falso)
    - 0 AND 1 = 0 (falso E verdadeiro = falso)  
    - 1 AND 0 = 0 (verdadeiro E falso = falso)
    - 1 AND 1 = 1 (verdadeiro E verdadeiro = verdadeiro)
    
    O neurônio vai aprender automaticamente os pesos corretos!
    """
    # Cria neurônio com 2 entradas
    neuronio = Neuronio(2)
    
    # Dados: função AND
    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 0, 0, 1]
    
    print("🧠 Treinando neurônio para função AND...")
    
    # Treina 1000 vezes
    for epoca in range(1000):
        for i in range(4):
            neuronio.treinar(X[i], y[i])
    
    # Testa
    print("\n📊 Resultados:")
    for i in range(4):
        resultado = neuronio.forward(X[i])
        print(f"{X[i][0]} AND {X[i][1]} = {resultado:.3f} ({'✅' if (resultado > 0.5) == y[i] else '❌'})")
    
    print(f"\n⚙️ Pesos: {neuronio.pesos}")
    print(f"⚙️ Bias: {neuronio.bias:.3f}")