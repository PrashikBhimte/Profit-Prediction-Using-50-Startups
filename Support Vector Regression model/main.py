import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

y_train = y_train.reshape(len(y_train), -1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

X_train = sc_x.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train).reshape(len(y_train))
X_test = sc_x.transform(X_test)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

y_pred = sc_y.inverse_transform(regressor.predict(X_test).reshape(-1, 1))
y_pred = y_pred.reshape(len(y_pred))

X_axis = ['case' + str(i) for i in range(1, 11)]

X_axis_len = np.arange(len(X_axis))

plt.bar(X_axis_len - 0.2, y_test, 0.4, label = 'True', color = 'blue')
plt.bar(X_axis_len + 0.2, y_pred, 0.4, label = 'Predicated', color = 'red')

plt.xticks(X_axis_len, X_axis)
plt.xlabel("Cases")
plt.ylabel("Total Profit in Rs.")
plt.title("Support Vector Regression")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
