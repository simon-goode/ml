from scipy.io import loadmat
import numpy as np
from model import neural_network
from randinitialize import initialize
from prediction import predict
from scipy.optimize import minimize

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


m = X.shape[0]
input_layer_size = 784  # images are 28*28, so 784 features
hidden_layer_size = 100
num_labels = 10

# randomly initializing thetas
initial_theta1 = initialize(hidden_layer_size, input_layer_size)
initial_theta2 = initialize(num_labels, hidden_layer_size)

# unrolling parameters into a single column vector
initial_nn_params = np.concatenate((initial_theta1.flatten(), initial_theta2.flatten()))
maxiter = 100
lambda_reg = 0.1 # to avoid overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# calling minimize function to minimize cost function and train weights
results = minimize(neural_network, x0=initial_nn_params, args=myargs,
                   options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)
nn_params = results["x"] # trained theta is extracted

theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1)) # shape = (100, 785)
theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1)) # shape = (10, 101)

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

# saving thetas in .txt file
np.savetxt('theta1.txt', theta1, delimiter=' ')
np.savetxt('theta2.txt', theta2, delimiter=' ')