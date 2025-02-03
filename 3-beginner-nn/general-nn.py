import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        np.random.seed(61)
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))
        self.l = []
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward propagation
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = sigmoid(hidden_layer_input)
            
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output = sigmoid(output_layer_input)
            
            # Store outputs for plotting
            self.l.append(output.copy())
            
            # Compute error
            error = y - output
            
            # Backpropagation
            output_gradient = error * d_sigmoid(output)
            hidden_layer_error = output_gradient.dot(self.weights_hidden_output.T)
            hidden_gradient = hidden_layer_error * d_sigmoid(hidden_layer_output)
            
            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_gradient) * self.learning_rate
            self.weights_input_hidden += X.T.dot(hidden_gradient) * self.learning_rate
            self.bias_output += np.sum(output_gradient, axis=0, keepdims=True) * self.learning_rate
            self.bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * self.learning_rate
    
    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        return sigmoid(output_layer_input)

    def plot_convergence(self):
        self.l = np.array(self.l)
        plt.plot(self.l[:,:,0])
        plt.xlabel("Epochs")
        plt.ylabel("Output")
        plt.title("Convergence of Neural Network Outputs")
        plt.show()


# Training data for XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y)

# Predict outputs after training
predictions = nn.predict(X)
print("Predicted outputs after training:")
print(predictions)

# Plot convergence
nn.plot_convergence()