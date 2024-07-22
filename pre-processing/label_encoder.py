from sklearn.preprocessing import LabelEncoder

# Exemplo de dados categóricos
data = ['A', 'B', 'A', 'C']
encoder = LabelEncoder()
encoded = encoder.fit_transform(data)

print(f"Encoded --> {encoded}")
print(f"encoder.inverse_transform(encoded) --> {encoder.inverse_transform(encoded)}")