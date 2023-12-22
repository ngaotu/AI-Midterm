import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn import datasets, linear_model

advertising = pd.DataFrame(pd.read_csv("./advertising.csv"))
print("Head of the list:",advertising.head())
print("Rows and Columns:",advertising.shape)
print(advertising.info())

# Let's see how Sales are related with other variables using scatter plot.
sb.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

# Let's see the correlation between different variables.
sb.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()

# Convert data to array in Numpy
X = advertising['TV'].to_numpy()
y = advertising['Sales'].to_numpy()
X = np.array([X]).T
y = np.array([y]).T


one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w.T)

w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 300, 2)
y0 = w_0 + w_1*x0
# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([0, 300, 0, 25])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

# Checking with R squared
def mean(x):
    return sum(x)/len(x)
yhat = []
for i in X:
    yhat.append(w_0 + w_1*i[0])
a = b = j = 0
d = []
for i in y:
    d.append(i[0])
m = float(mean(d))
for i in d:
    a = a + (i-m)**2
    b = b + (i-yhat[j])**2
    j = j+1
Rsquared = 1 - b/a
print("R squared =",Rsquared)

# Using Scikit-Learn Library
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by formula: ', w.T)