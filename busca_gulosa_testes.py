from Grafo import Grafo
from BuscaGulosa import Gulosa

grafo = Grafo()

busca_gulosa = Gulosa(grafo.bucharest)
busca_gulosa.buscar(grafo.arad)
