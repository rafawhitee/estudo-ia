import os
import pandas as pd

df_credit_risk = pd.read_csv(os.getcwd() + '\\pre_processing\\curso_ia_academy\\credit_risk_dataset.csv')
print(f"{df_credit_risk}")