import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#оказались не самые полезные графики))
df = pd.read_csv('../data/avito_bmw_cars.csv')

numeric_columns = ['Year', 'Mileage', 'Engine_capacity', 'Power', 'Fuel_consumption_mixed', 'Acceleration_to_100', 'Cost']
df[numeric_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Гистограммы числовых переменных')
plt.show()

sns.pairplot(df[numeric_columns])
plt.suptitle('Парные графики для числовых переменных', y=1.02)
plt.show()

correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.show()
plt.close('all')
