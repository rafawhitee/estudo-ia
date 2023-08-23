import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

source = pd.read_csv('cars.csv')
sns.scatterplot(x="mileage_per_year", y="price", hue="sold", data=source)
plt.show()