import pandas as pd
import matplotlib.pyplot as plt

unrate = pd.read_csv('data/unrate.csv')
unrate['DATE'] = pd.to_datetime(unrate['DATE'])
x_values = unrate.head(12)['DATE']
y_values = unrate.head(12)['VALUE']

plt.plot(x_values, y_values)
plt.xticks(rotation=90)
plt.xlabel('Month')
plt.ylabel('Unemployment Rate')
plt.title("Monthly Unemployment Trends, 1948")
plt.show()
