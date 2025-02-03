import numpy as np
from scipy.io import loadmat
from prediction import predict

theta1 = np.loadtxt('theta1.txt')
theta2 = np.loadtxt('theta2.txt')

# loading mat file
data = loadmat('mnist-original.mat')

# extracting features from mat file
X = data['data']
X = X.transpose()

# normalizing features
X = X / 255

# extracting labels from mat file
y = data['label']
y = y.flatten()

# training set, 60000 samples
X_train = X[:60000, :]
y_train = y[:60000]

# validation set, 10000 samples
X_test = X[60000:, :]
y_test = y[60000:]

# check validation set accuracy of model
pred = predict(theta1, theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))

# check training set accuracy of model
pred = predict(theta1, theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))

# precision eval
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision = ', true_positive/(true_positive + false_positive))