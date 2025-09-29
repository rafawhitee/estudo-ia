import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, cityblock, euclidean

def distancia_euclidiana(v1, v2):
    """
    Distância Euclidiana - a "linha reta" entre dois pontos
    Fórmula: sqrt(sum((v1_i - v2_i)²))
    """
    return np.sqrt(np.sum((v1 - v2) ** 2))

def distancia_manhattan(v1, v2):
    """
    Distância Manhattan (L1) - soma das diferenças absolutas
    Fórmula: sum(|v1_i - v2_i|)
    Útil quando mudanças em dimensões são independentes
    """
    return np.sum(np.abs(v1 - v2))

def distancia_cosseno(v1, v2):
    """
    Distância do Cosseno - baseada no ângulo entre vetores
    Fórmula: 1 - (v1·v2) / (||v1|| * ||v2||)
    Muito usada em NLP para similaridade de documentos
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return 1 - cosine_similarity

def similaridade_cosseno(v1, v2):
    """
    Similaridade do Cosseno (complemento da distância)
    Retorna valor entre -1 e 1
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Exemplos práticos
if __name__ == "__main__":
    # Vetores de exemplo
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.array([2, 4, 6])  # v3 é proporcional a v1
    
    print("=== COMPARAÇÃO DE DISTÂNCIAS ===")
    print(f"Vetor 1: {v1}")
    print(f"Vetor 2: {v2}")
    print(f"Vetor 3: {v3}")
    print()
    
    # Calculando distâncias v1 vs v2
    print("Distâncias entre v1 e v2:")
    print(f"Euclidiana: {distancia_euclidiana(v1, v2):.4f}")
    print(f"Manhattan:  {distancia_manhattan(v1, v2):.4f}")
    print(f"Cosseno:    {distancia_cosseno(v1, v2):.4f}")
    print(f"Similaridade Cosseno: {similaridade_cosseno(v1, v2):.4f}")
    print()
    
    # Calculando distâncias v1 vs v3 (proporcionais)
    print("Distâncias entre v1 e v3 (v3 = 2*v1):")
    print(f"Euclidiana: {distancia_euclidiana(v1, v3):.4f}")
    print(f"Manhattan:  {distancia_manhattan(v1, v3):.4f}")
    print(f"Cosseno:    {distancia_cosseno(v1, v3):.4f}")
    print(f"Similaridade Cosseno: {similaridade_cosseno(v1, v3):.4f}")
    print()
    
    # Exemplo prático em NLP
    print("=== EXEMPLO NLP - SIMILARIDADE DE DOCUMENTOS ===")
    # Representação bag-of-words simplificada
    # Dimensões: [python, machine, learning, data, science]
    doc1 = np.array([3, 2, 1, 0, 1])  # "python python python machine machine learning science"
    doc2 = np.array([1, 3, 2, 2, 1])  # "python machine machine machine learning learning data data science"
    doc3 = np.array([0, 0, 0, 5, 3])  # "data data data data data science science science"
    
    print("Documentos representados como vetores:")
    print(f"Doc 1 (foco em Python): {doc1}")
    print(f"Doc 2 (foco em ML):     {doc2}")
    print(f"Doc 3 (foco em Data):   {doc3}")
    print()
    
    print("Similaridades entre documentos (cosseno):")
    print(f"Doc1 vs Doc2: {similaridade_cosseno(doc1, doc2):.4f}")
    print(f"Doc1 vs Doc3: {similaridade_cosseno(doc1, doc3):.4f}")
    print(f"Doc2 vs Doc3: {similaridade_cosseno(doc2, doc3):.4f}")
    print()
    
    # Exemplo com vetores 2D para visualização
    print("=== VISUALIZAÇÃO 2D ===")
    # Vetores 2D para plotar
    p1 = np.array([1, 1])
    p2 = np.array([4, 2])
    p3 = np.array([2, 4])
    
    plt.figure(figsize=(10, 6))
    
    # Plot dos pontos
    plt.scatter(*p1, color='red', s=100, label='P1 (1,1)')
    plt.scatter(*p2, color='blue', s=100, label='P2 (4,2)')
    plt.scatter(*p3, color='green', s=100, label='P3 (2,4)')
    
    # Linhas conectando os pontos
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.7, label='P1-P2')
    plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'g--', alpha=0.7, label='P1-P3')
    
    # Configurações do plot
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Visualização das Distâncias')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Anotações com as distâncias
    plt.annotate(f'Euclidiana: {distancia_euclidiana(p1, p2):.2f}', 
                xy=((p1[0]+p2[0])/2, (p1[1]+p2[1])/2), xytext=(10, 10),
                textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print("Comparação das distâncias P1-P2 vs P1-P3:")
    print(f"P1-P2 | Euclidiana: {distancia_euclidiana(p1, p2):.4f}, Manhattan: {distancia_manhattan(p1, p2):.4f}")
    print(f"P1-P3 | Euclidiana: {distancia_euclidiana(p1, p3):.4f}, Manhattan: {distancia_manhattan(p1, p3):.4f}")
    
    # Verificação com scipy
    print("\n=== VERIFICAÇÃO COM SCIPY ===")
    print(f"Euclidiana (scipy): {euclidean(v1, v2):.4f}")
    print(f"Manhattan (scipy):  {cityblock(v1, v2):.4f}")
    print(f"Cosseno (scipy):    {cosine(v1, v2):.4f}")