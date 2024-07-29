from service import get_embeddings, get_chat_completion

pergunta = "qual a derivada de 4x³ + 6x² -5x + 100 que tem reta tangente no ponto -2 ?"
#print(f"'{pergunta}' em vetor matemático Emmbeddings -> {get_embeddings(pergunta)} \n")
print(f"Resposta do ChatGPT para '{pergunta}' --> {get_chat_completion(pergunta)} \n")