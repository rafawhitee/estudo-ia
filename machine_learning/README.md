# Machine Learning

## 🧠 Perceptron - O Ancestral das Redes Neurais

O **perceptron** é usado como bloco fundamental em ML, especialmente em:

### 📚 **O que é o Perceptron?**
- **Inventor**: Frank Rosenblatt (1957)
- **Conceito**: Primeiro modelo matemático de neurônio artificial
- **Função**: Classificação binária (sim/não, 0/1)

### 🔧 **Onde é usado hoje?**

1. **Redes Neurais Modernas**
   - Cada neurônio em uma rede neural é basicamente um perceptron
   - Deep Learning = milhões de perceptrons conectados

2. **Algoritmos Clássicos**
   - **SVM (Support Vector Machine)**: evolução do perceptron
   - **Regressão Logística**: versão probabilística do perceptron
   - **Linear Classifiers**: família de algoritmos baseados no perceptron

3. **Aplicações Práticas**
   - Reconhecimento de imagens (camadas de perceptrons)
   - Processamento de linguagem natural
   - Sistemas de recomendação
   - Detecção de spam

### ⚡ **Perceptron vs Neurônio Artificial**

```python
# São praticamente a mesma coisa!
# Diferença: função de ativação

# Perceptron clássico (função degrau)
def perceptron_ativacao(z):
    return 1 if z >= 0 else 0

# Neurônio moderno (função sigmoid)  
def neuronio_ativacao(z):
    return 1 / (1 + exp(-z))
```

### 🎯 **Limitações Históricas**
- **Problema XOR**: Um perceptron não consegue resolver XOR
- **Solução**: Perceptron Multicamadas (MLP) = Redes Neurais

### 📁 **Arquivos neste diretório**
- `01_neuronio.py` - Implementação básica (versão moderna do perceptron)
- Próximos: MLP, backpropagation, deep learning...

**Resumo**: O perceptron é o "DNA" do Machine Learning moderno! 🧬