import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key=os.getenv("OPENAI_API_KEY")

default_model = "gpt-4"
default_model_embedding = "text-embedding-ada-002"

# calcula similiarinade entre 2 vetores matemáticos
def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

# pega o conteúdo (texto) e faz o embedding para um vetor matemático
def get_embeddings(content: str, model: str = default_model_embedding):
    response = openai.Embedding.create(input=content,model=model)
    return response['data'][0]['embedding']

# pega os embeddings (vetor matemático) e transforma em string
def embedding_to_text(embedding: str):
    # Converte o vetor de embedding em uma string de texto para incluir no prompt
    return ' '.join(map(str, embedding))

# recebe um conteúdo (texto) e chama a api do chatgpt para ele retornar a resposta da IA
def get_chat_completion(content:str, model:str = default_model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": content}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

texto_1 = "Explique a teoria da relatividade de forma simples."
texto_2 = "Descreva a teoria da relatividade em termos fáceis."

print(f"Similaridade coseno entre os textos: {calculate_cosine_similarity(get_embeddings(texto_1), get_embeddings(texto_2))} \n")

pergunta = "qual a derivada de 4x³ + 6x² -5x + 100 que tem reta tangente no ponto -2?"
#print(f"'{pergunta}' em vetor matemático Emmbeddings -> {get_embeddings(pergunta)} \n")
print(f"Resposta do ChatGPT para '{pergunta}' -> {get_chat_completion(pergunta)} \n")