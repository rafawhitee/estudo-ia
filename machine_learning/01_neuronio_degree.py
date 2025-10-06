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
    
    def step_function(self, z):
        """
        Função de ativação degrau (step function) - PERCEPTRON CLÁSSICO
        
        Exemplos:
        - step_function(-2) = 0  (qualquer valor negativo = 0)
        - step_function(0)  = 1  (zero ou positivo = 1)
        - step_function(5)  = 1  (qualquer valor positivo = 1)
        
        É como um "interruptor digital": ou dispara (1) ou não dispara (0).
        Não há meio termo como no sigmoid!
        """
        return 1 if z >= 0 else 0
    
    def forward(self, entradas):
        """
        Propagação para frente: calcula a saída do neurônio.
        
        Passos:
        1. Soma ponderada: z = entrada1*peso1 + entrada2*peso2 + bias
        2. Ativação: saída = step_function(z) -> 0 ou 1
        
        Exemplo com entradas=[1, 0], pesos=[0.5, -0.3], bias=0.1:
        - z = 1*0.5 + 0*(-0.3) + 0.1 = 0.6
        - saída = step_function(0.6) = 1 (porque 0.6 >= 0)
        """
        # Converte para array NumPy se necessário
        entradas = np.array(entradas)
        # Soma ponderada + ativação
        z = np.dot(entradas, self.pesos) + self.bias
        return self.step_function(z)
    
    def treinar(self, entradas, target, lr=0.5):
        """
        Treina o neurônio ajustando pesos baseado no erro - REGRA DO PERCEPTRON
        
        Algoritmo (Perceptron Learning Rule):
        1. Faz predição (0 ou 1)
        2. Calcula erro = target - predição  
        3. Se erro != 0, ajusta pesos
        4. Novo_peso = peso_atual + lr * erro * entrada
        
        Exemplo: target=1, predição=0, entrada=[1,0]
        - erro = 1 - 0 = 1 (erro positivo)
        - peso1 += 0.5 * 1 * 1 = +0.5 (aumenta peso da entrada ativa)
        - peso2 += 0.5 * 1 * 0 = 0 (não muda peso da entrada inativa)
        """
        # Converte para array NumPy
        entradas = np.array(entradas)
        
        # Predição
        predicao = self.forward(entradas)
        
        # Erro (só pode ser -1, 0 ou +1)
        erro = target - predicao
        
        # Atualiza pesos APENAS se houver erro (regra clássica do perceptron)
        if erro != 0:
            self.pesos += lr * erro * entradas
            self.bias += lr * erro
        
        return erro

# Exemplo: Neurônio aprendendo função AND
if __name__ == "__main__":
    """
    Demonstração: ensina um perceptron a simular porta lógica AND
    
    Função AND:
    - 0 AND 0 = 0 (falso E falso = falso)
    - 0 AND 1 = 0 (falso E verdadeiro = falso)  
    - 1 AND 0 = 0 (verdadeiro E falso = falso)
    - 1 AND 1 = 1 (verdadeiro E verdadeiro = verdadeiro)
    
    O perceptron vai aprender automaticamente os pesos corretos!
    Diferença: saída será exatamente 0 ou 1 (não probabilidades)
    """
    # Cria neurônio com 2 entradas
    neuronio = Neuronio(2)
    
    # Dados: função AND
    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 0, 0, 1]
    
    print("🧠 Treinando PERCEPTRON para função AND...")
    
    # Treina até convergir (ou máximo 1000 épocas)
    for epoca in range(1000):
        erro_total = 0
        for i in range(4):
            erro = neuronio.treinar(X[i], y[i])
            erro_total += abs(erro)
        
        # Se não há mais erros, convergiu!
        if erro_total == 0:
            print(f"✅ Convergiu na época {epoca + 1}!")
            break
    
    # Testa
    print("\n📊 Resultados:")
    for i in range(4):
        resultado = neuronio.forward(X[i])
        print(f"{X[i][0]} AND {X[i][1]} = {resultado} ({'✅' if resultado == y[i] else '❌'})")
    
    print(f"\n⚙️ Pesos: {neuronio.pesos}")
    print(f"⚙️ Bias: {neuronio.bias:.3f}")
    
    print(f"\n💡 PERCEPTRON vs NEURÔNIO MODERNO:")
    print(f"• Perceptron: saída binária (0 ou 1)")
    print(f"• Neurônio moderno: saída probabilística (0.0 a 1.0)")
    print(f"• Perceptron: função degrau")
    print(f"• Neurônio moderno: função sigmoid/relu/etc")