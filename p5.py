import numpy as np

# Input data normalization
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float) / 9
y = np.array([[92], [86], [89]], dtype=float) / 100

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Function
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network parameters
epochs = 7000
learning_rate = 0.1
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Weight and bias initialization
input_hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_biases = np.random.uniform(size=(1, hidden_neurons))
hidden_output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_biases = np.random.uniform(size=(1, output_neurons))

# Training loop
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, input_hidden_weights) + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, hidden_output_weights) + output_biases
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error_output = y - predicted_output
    output_gradient = sigmoid_derivative(predicted_output)
    d_output = error_output * output_gradient
    error_hidden = d_output.dot(hidden_output_weights.T)
    hidden_gradient = sigmoid_derivative(hidden_layer_output)
    d_hidden = error_hidden * hidden_gradient

    # Update weights and biases
    hidden_output_weights += hidden_layer_output.T.dot(d_output) * learning_rate
    input_hidden_weights += X.T.dot(d_hidden) * learning_rate

# Display results
print("Input: \n", X)
print("Actual Output: \n", y)
print("Predicted Output: \n", predicted_output)
