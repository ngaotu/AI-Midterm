from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import numpy as np


iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)


classifier = SVC(kernel = 'linear', decision_function_shape='ovr' ,random_state = 0)

classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)

y_pred = classifier.predict(X_test)
class_report = classification_report(y_test, y_pred)
w = classifier.coef_
b = classifier.intercept_
print('w = ', w)
print('b = ', b)
accuracy = np.mean(y_pred == y_test)
print(f"Độ chính xác: {accuracy}")
print("")
print(f'Classification report:\n{class_report}')



