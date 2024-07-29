import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from service import get_embeddings

# calcula similiarinade entre 2 vetores matemáticos
def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

texto_1 = "Explique a teoria da relatividade de forma simples."
texto_2 = "Descreva a teoria da relatividade em termos fáceis."

print(f"Similaridade coseno entre os textos --> {calculate_cosine_similarity(get_embeddings(texto_1), get_embeddings(texto_2))} \n")