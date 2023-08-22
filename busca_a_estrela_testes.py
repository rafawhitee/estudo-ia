from Grafo import Grafo
from BuscaAEstrela import AEstrela

grafo = Grafo()

busca_a_estrela = AEstrela(grafo.bucharest)
busca_a_estrela.buscar(grafo.arad)
