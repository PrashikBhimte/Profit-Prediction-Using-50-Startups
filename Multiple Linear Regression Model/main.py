import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

X_axis = ['case' + str(i) for i in range(1, 11)]

X_axis_len = np.arange(len(X_axis))

plt.bar(X_axis_len - 0.2, Y_test, 0.4, label = 'True', color = 'blue')
plt.bar(X_axis_len + 0.2, Y_pred, 0.4, label = 'Predicated', color = 'red')

plt.xticks(X_axis_len, X_axis)
plt.xlabel("Cases")
plt.ylabel("Total Profit in Rs.")
plt.title("Multiple Linear Regression")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))
