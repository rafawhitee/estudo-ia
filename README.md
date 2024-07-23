# ::computer:: Estudo de Inteligência Artificial

## ChatGPT

## Deep Learning

## Lógica Fuzzy

## Machine Learning

### Curso Alura 001
https://cursos.alura.com.br/course/machine-learning-introducao-a-classificacao-com-sklearn

### Curso Alura 002
https://cursos.alura.com.br/course/machine-learning-classificacao-tras-panos

### Curso Alura 003
https://cursos.alura.com.br/course/machine-learning-validando-modelos

#### Primeira Parte (Cross Validate)
- Para não ter que tomar decisões baseados em uma única execução do código de Machine Learning (que envolve aleatoriedade), entra a Validação Cruzada.

- Com a validação cruzada você divide os dados em partes e roda o algoritmo N vezes, e vez que roda, o algoritmo terá uma nova "população" de treino e de teste.
```python
resultado = cross_validate(modelo, x, y, cv = 5) # o cv é o parâmetro onde você passa em quantas vezes você deseja quebrar (e consequentemente rodar os treinamentos/testes)
```


#### Segunda Parte (Cross Validate com Aleatoriedade)
- No código da primeira parte, a gente usou o modelo "Árvore de Decisão" junto com o Cross Validate, mas para deixar o código melhor, existe uma maneira de fazer uma aleatoriedade 
nos dados principal, imagina você pegar seu dataset e embaralhar toda vez antes de rodar o Cross Validate.

- No parâmetro cv do método cross_validate, ele além de aceitar um número, aceita passar o objeto KFold, onde você pode alterar para ele embaralhar (shuffle = True).
```python
cv = KFold(n_splits = 10, shuffle = True)
resultado = cross_validate(modelo, x, y, cv = cv)
```

#### Terceira Parte (Estratificação)
- No código da segunda parte, a gente usou o modelo_selecion KFold para o Cross Validate, mas dependendo, suas classes pode ter bastante diferença em quantidade.

- Por exemplo, se você pode ter 4 saídas (classes), a primeira classe tem 1002 registros, a segunda 800, terceira 700 e a quarta 50, a diferença é brusca.

- Então nesses casos que uma das classes estão muito distantes, existe a Estratificaçao para ajudar no código.
```python
cv = StratifiedKFold(n_splits = 10, shuffle = True)
resultado = cross_validate(modelo, x, y, cv = cv)
```

#### Quarta Parte (Agrupador)
- Digamos agora que nossos dados possuem colunas que são agrupadoras (no caso do exemplo, a gente criou uma coluna Modelo de Carro populada de forma aleatória)

- Se você tem um agrupador no seu dataset, não é UMA BOA ESCOLHA o Algoritmo em questão treinar/testar com todos os modelos, pois digamos que você entre um novo modelo,
o algoritmo iria conseguir prever?

Gerando a nova coluna com base no ano do carro
```python
dados["modelo_carro_aleatorio"] = dados.idade_do_modelo + np.random.randint(-2, 3, size=10000)
dados["modelo_carro_aleatorio"] = dados["modelo_carro_aleatorio"] + abs(dados["modelo_carro_aleatorio"].min()) + 1
```

Usando o Modelo GroupKFold que trata essa questão de agrupamento
```python
cv = GroupKFold(n_splits = 10)
resultado = cross_validate(modelo, x, y, cv = cv, groups=dados["modelo_carro_aleatorio"])
```

## Otimização