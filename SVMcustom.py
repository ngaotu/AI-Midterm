import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

class SVM:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (1/(epoch+1) * self.weights)
                else:
                    self.weights -= self.lr * (1/(epoch+1) * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.lr * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)

def train_ova_svm(X_train, y_train):
    classes = np.unique(y_train)
    models = {}

    for i in classes:
        binary_y = np.where(y_train == i, 1, -1)
        model = SVM()
        model.train(X_train, binary_y)
        models[i] = (model.weights, model.bias)

    return models

def predict_ova_svm(X, models):
    scores = np.zeros((X.shape[0], len(models)))

    for i, (weights, bias) in models.items():
        scores[:, i] = np.dot(X, weights) - bias

    return np.argmax(scores, axis=1)

# Train OVA SVM
models = train_ova_svm(X_train, y_train)

# Make predictions
y_pred = predict_ova_svm(X_test, models)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Sentosa', 'Versicolor','Virginica'])
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f'Classification report:\n{class_report}')
# Display weights and bias for each class
for i, (weights, bias) in models.items():
    print(f"Class {i} - Weights: {weights}, Bias: {bias}")
    
print("Confusion Matrix:")
cm_display.plot()
plt.show()
