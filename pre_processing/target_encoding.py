import pandas as pd

# Dados e target originais
df = pd.DataFrame({'Feature': ['A', 'B', 'A', 'C'], 'Target': [1, 0, 1, 0]})

# Aplicando Target Encoding
target_mean = df.groupby('Feature')['Target'].mean()
df['Feature_encoded'] = df['Feature'].map(target_mean)

# Revertendo a transformação usando o mapeamento original
inverse_map = {v: k for k, v in target_mean.to_dict().items()}
df['Feature_inversed'] = df['Feature_encoded'].map(inverse_map)

print(df)