import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Entrada - ângulo (-90 a 90 graus)
angulo = ctrl.Antecedent(np.arange(-90, 91, 1), 'angulo')
angulo.automf(number=5, names=['pos_big', 'pos_small', 'zero', 'neg_small', 'neg_big'])

# Entrada - distância (-10 a 30 jardas)
distancia = ctrl.Antecedent(np.arange(-10, 31, 1), 'distancia')
distancia.automf(number=5, names=['neg_close', 'zero', 'close', 'medium', 'far'])

# Saída - potência (-30 a 30 kW)
potencia = ctrl.Consequent(np.arange(-30, 31, 1), 'potencia')

potencia['neg_high'] = fuzz.trimf(potencia.universe, [-30, -25, -10])
potencia['neg_medium'] = fuzz.trimf(potencia.universe, [-25, -10, 0])
potencia['zero'] = fuzz.trimf(potencia.universe, [-10, 0, 10])
potencia['pos_medium'] = fuzz.trimf(potencia.universe, [0, 10, 23])
potencia['pos_high'] = fuzz.trimf(potencia.universe, [10, 23, 30])


# Define as Regras
far1 = ctrl.Rule(distancia['far'] & angulo['neg_big'], potencia['pos_high'])
far2 = ctrl.Rule(distancia['far'] & angulo['neg_small'], potencia['pos_high'])
far3 = ctrl.Rule(distancia['far'] & angulo['zero'], potencia['pos_high'])
far4 = ctrl.Rule(distancia['far'] & angulo['pos_small'], potencia['pos_high'])
far5 = ctrl.Rule(distancia['far'] & angulo['pos_big'], potencia['pos_medium'])

medium1 = ctrl.Rule(distancia['medium'] & angulo['neg_big'], potencia['pos_high'])
medium2 = ctrl.Rule(distancia['medium'] & angulo['neg_small'], potencia['pos_medium'])
medium3 = ctrl.Rule(distancia['medium'] & angulo['zero'], potencia['pos_medium'])
medium4 = ctrl.Rule(distancia['medium'] & angulo['pos_small'], potencia['zero'])
medium5 = ctrl.Rule(distancia['medium'] & angulo['pos_big'], potencia['zero'])

close1 = ctrl.Rule(distancia['close'] & angulo['neg_big'], potencia['zero'])
close2 = ctrl.Rule(distancia['close'] & angulo['neg_small'], potencia['pos_high'])
close3 = ctrl.Rule(distancia['close'] & angulo['zero'], potencia['pos_high'])
close4 = ctrl.Rule(distancia['close'] & angulo['pos_small'], potencia['pos_high'])
close5 = ctrl.Rule(distancia['close'] & angulo['pos_big'], potencia['neg_medium'])

zero1 = ctrl.Rule(distancia['zero'] & angulo['neg_big'], potencia['pos_medium'])
zero2 = ctrl.Rule(distancia['zero'] & angulo['neg_small'], potencia['pos_medium'])
zero3 = ctrl.Rule(distancia['zero'] & angulo['zero'], potencia['zero'])
zero4 = ctrl.Rule(distancia['zero'] & angulo['pos_small'], potencia['neg_medium'])
zero5 = ctrl.Rule(distancia['zero'] & angulo['pos_big'], potencia['neg_medium'])

neg_close1 = ctrl.Rule(distancia['neg_close'] & angulo['neg_big'], potencia['pos_high'])
neg_close2 = ctrl.Rule(distancia['neg_close'] & angulo['neg_small'], potencia['pos_high'])
neg_close3 = ctrl.Rule(distancia['neg_close'] & angulo['zero'], potencia['neg_medium'])
neg_close4 = ctrl.Rule(distancia['neg_close'] & angulo['pos_small'], potencia['neg_high'])
neg_close5 = ctrl.Rule(distancia['neg_close'] & angulo['pos_big'], potencia['neg_high'])

# Sistema de Controle (TESTES)

# gera um controlSystem, com as regras
sistema_controle = ctrl.ControlSystem([far1, far2, far3, far4, far5, medium1, medium2, medium3, medium4, medium5, close1, close2, close3, close4, close5,
    zero1, zero2, zero3, zero4, zero5, neg_close1, neg_close2, neg_close3, neg_close4, neg_close5])

# cria um sistema simulado com o controle acima
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# insere as entradas (no caso, angulo e distancia)
sistema.input['angulo'] = -90
sistema.input['distancia'] = -10

# executa
sistema.compute()

# printa a saida (no caso a potência que o motor deve fazer)
print(sistema.output['potencia'])
potencia.view(sim = sistema)

plt.show()