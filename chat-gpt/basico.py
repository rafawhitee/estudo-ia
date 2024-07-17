import os
import openai

openai.api_key=os.getenv("OPENAI_API_KEY")

default_model = "gpt-4"

def get_embeddings(content: str, model: str = default_model):
    response = openai.Embedding.create(input=content,model=model)
    return response['data'][0]['embedding']

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

print(f"Response Emmbeddings -> {get_embeddings("Quanto é 5x5+4?", model="text-embedding-ada-002")}")
print(f"Response ChatCompletion -> {get_chat_completion("Quanto é 5x5+4?")}")