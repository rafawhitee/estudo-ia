# Projeto utilizando SKLearn do Python para adivinhar por Machine Learning se é 'Porco' ou 'Cachorro'

### Cachorro ou Porco
Machine Learning para fazer o predict a partir de algumas features, se é 'Porco' ou 'Cachorro'

### Tracking
Machine Learning para fazer o predict a partir de alguns dados de tracking de um website, sendo que cada linha representa um usuário e quais sites ele acessou.

O resultado esperado (o Y) é se ele comprou algo, ou seja, a partir do acesso em alguns dados, quer fazer uma prediction se ele irá comprar

### Projects
Com o Machine Learning tentamos usar o LinearSVC para fazer estimativas, mas além da acurácia ser baixa, o gráfico da estimativa não ficou bom (pois ele é linear).

E se repararmos no gráfico inicial dos dados dos projetos, ele forma uma curva, então precisamos usar o SVC (que não é linear).

Só que o SVC tem um pequeno problema: a diferença de escala é muito grande, no Eixo X ia de 0 até 1, mas no Y ia de 0 até 30.000.

Então além do SVC, utilizamos também o StandardScaler que ele muda as escalas para deixar mais próximos.
