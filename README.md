# Estudo de Inteligência Artificial

## Base Matemática

📌 1. Fundamentos

a. Vetores

- Operações básicas: soma, subtração, produto escalar, norma (magnitude).

- Distâncias: Euclidiana, Manhattan, cosseno (muito usadas em ML/NLP).

- Interpretação geométrica: vetores como pontos e direções no espaço.

b. Matrizes

- Operações: adição, multiplicação, transposição.

- Matrizes identidade e diagonais.

- Multiplicação por vetor (combinações lineares).

- Matrizes como transformações lineares (rotação, escala, projeção).

📌 2. Conceitos Essenciais para ML/DL

- Produto Matriz-Vetor e Matriz-Matriz

- Base da multiplicação em redes neurais (camadas são essencialmente 
𝑊
⋅
𝑥
+
𝑏
W⋅x+b).

- Determinante e Posto (Rank)

- Determinante → quando uma matriz é invertível.

- Rank → redundância de dados, dimensão de subespaço (importante em redução de dimensionalidade).

- Autovalores e Autovetores

- Fundamentais em PCA (redução de dimensionalidade).

- Intuição: direções de maior variação nos dados.

- Decomposições (avançando um pouco)

- SVD (Singular Value Decomposition) → base de PCA, compressão de dados, embeddings.

- Eigendecomposição → análise de matrizes simétricas (ex.: covariância).

📌 3. Aplicações em Machine Learning

- Regressão Linear: solução com álgebra matricial (equações normais).

- Gradiente: cálculo de derivadas em espaços vetoriais.

- PCA: autovalores/autovetores da matriz de covariância.

- Redes Neurais: cada camada é multiplicação de matriz + função de ativação.

- Distâncias e similaridades: medidas vetoriais para clustering, embeddings, NLP.

📌 4. O que é mais útil focar primeiro?

- Vetores e matrizes (operação e interpretação).

- Produto escalar, norma, distância e ângulo entre vetores.

- Multiplicação de matrizes → entender camadas de rede.

- Autovalores/autovetores → PCA e compressão.

- SVD → se for avançar para recomendações, NLP ou visão computacional.