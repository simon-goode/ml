import numpy as np

def predict(theta1, theta2, X):
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1) # adding bias unit to first layer
    z2 = np.dot(X, theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2)) # activation of first layer
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1) # adding bias unit to hidden layer
    z3 = np.dot(a2, theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3)) # activation of third layer
    p = (np.argmax(a3, axis=1)) # predicting class based on max value of hypothesis

    return p