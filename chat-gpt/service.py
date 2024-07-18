import os
import openai

openai.api_key=os.getenv("OPENAI_API_KEY")

default_model = "gpt-4"
default_model_embedding = "text-embedding-ada-002"

# pega o conteúdo (texto) e faz o embedding para um vetor matemático
def get_embeddings(content: str, model: str = default_model_embedding):
    response = openai.Embedding.create(input=content,model=model)
    return response['data'][0]['embedding']

# recebe um conteúdo (texto) e chama a api do chatgpt para ele retornar a resposta da IA
def get_chat_completion(content:str, max_tokens:int=150, temperature:float=0.7,model:str = default_model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": content}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message['content'].strip()