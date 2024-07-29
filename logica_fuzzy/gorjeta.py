import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Declarando as Entradas (ambos são da escala de 0 a 10)
qualidade = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')

# Declarando a Saída (Gorjeta) - 0% a 20%
gorjeta = ctrl.Consequent(np.arange(0, 21, 1), 'gorjeta')


# Faz o mapeamento das entradas para as categorias (Classes)
qualidade.automf(number=3, names=['ruim', 'boa', 'saborosa'])
servico.automf(number=3, names=['ruim', 'aceitavel', 'otimo'])

# Mostra visualmente os gráficos gerados pela Lógica Fuzzy para as Entradas
qualidade.view()
servico.view()

# Define quais os valores para gorjeta baixa, média e alta

## baixa começará no 0, o pico é 0 e vai até o 8
gorjeta['baixa'] = fuzz.trimf(gorjeta.universe, [0, 0, 8])

## media começará no 2, o pico é no 10 e vai até o 18
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [2, 10, 18])

## alta começará no 12, o pico é 18 e vai até o 20
gorjeta['alta'] = fuzz.trimf(gorjeta.universe, [12, 18, 20])


# Define as Regras
regra1 = ctrl.Rule(qualidade['ruim'] | servico['ruim'], gorjeta['baixa'])
regra2 = ctrl.Rule(servico['aceitavel'], gorjeta['media'])
regra3 = ctrl.Rule(servico['otimo'] | qualidade['saborosa'], gorjeta['alta'])



# Sistema de Controle (TESTES)

# gera um controlSystem, com as regras
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])

# cria um sistema simulado com o controle acima
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# insere as entradas (no caso, qualidade e servico)
sistema.input['qualidade'] = 10
sistema.input['servico'] = 2

# executa
sistema.compute()

# printa a saida (no caso gorjeta)
print(sistema.output['gorjeta'])
gorjeta.view(sim = sistema)

plt.show()