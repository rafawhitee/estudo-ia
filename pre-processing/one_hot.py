from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Exemplo de dados categóricos
df = pd.DataFrame({'Feature': ['A', 'B', 'A', 'C']})

encoder = OneHotEncoder()
transformed = encoder.fit_transform(df[['Feature']])
df_transformed = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out())
print(f"Transformed --> {df_transformed}")

# Revertendo a transformação
inversed = encoder.inverse_transform(transformed)
df_inversed = pd.DataFrame(inversed, columns=['Feature'])
print(f"Inversed --> {df_inversed}")