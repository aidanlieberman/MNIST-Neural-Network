import sys
import numpy as np
import math
import random

class Network:
    def __init__ (self, layers):
        #layers is a list of layer instances, forming the MLP
        self.layers = layers

    def forward_pass_network (self, input_data):
        activations = input_data
        for layer in self.layers:
            activations = layer.forward_pass_layer(activations)
        return activations
        
    def backprop_network (self, first_upstream_gradient):
        upstream_gradient = first_upstream_gradient
        for layer in reversed(self.layers):
            upstream_gradient = layer.backprop_layer(upstream_gradient)
        
    def update_weights_network (self, learning_rate):
        for layer in self.layers:
            layer.update_weights_layer(learning_rate)

    def train (self, input_data, ideal_y):
        actual_y = self.forward_pass_network(input_data)
        first_upstream_grad = 2 * np.subtract(actual_y, ideal_y)
        self.backprop_network(first_upstream_grad)
        self.update_weights_network(0.1)


class Layer:
    def __init__ (self, weights):
        self.weights = weights
        self.z = np.array([])
        self.incoming_activations_hat = np.array([])
        self.grad_of_E_wrt_w = np.array([])

    def forward_pass_layer (self, incoming_activations):
        self.incoming_activations_hat = np.vstack((incoming_activations, [1]))
        self.z = np.matmul(self.weights, self.incoming_activations_hat)
        y = sigmoid(self.z)
        return y

    def backprop_layer (self, upstream_grad):
        h_prime_z = sigmoid_derivative(self.z)
        grad_of_E_wrt_z = np.multiply(h_prime_z, upstream_grad)
        t = self.incoming_activations_hat.transpose()
        self.grad_of_E_wrt_w = np.matmul(grad_of_E_wrt_z, t)
        w_t = self.weights.transpose()
        grad_of_E_wrt_xhat = np.matmul(w_t, grad_of_E_wrt_z)
        next_upstream_grad = np.delete(grad_of_E_wrt_xhat, grad_of_E_wrt_xhat.size - 1, 0)
        return next_upstream_grad

    def update_weights_layer (self, learning_rate):
        self.weights = np.subtract(self.weights, learning_rate * self.grad_of_E_wrt_w)

def sigmoid_unvectorized (element):
    return 1 / (1 + math.pow(math.e, (-1 * element)))  
sigmoid = np.vectorize(sigmoid_unvectorized)

def sigmoid_derivative_unvectorized (element):
    return sigmoid_unvectorized(element) * (1 - sigmoid_unvectorized(element))
sigmoid_derivative = np.vectorize(sigmoid_derivative_unvectorized)

weights1 = np.random.uniform(low=-0.001, high=0.001, size=(64, 785))
weights2 = np.random.uniform(low=-0.001, high=0.001, size=(64, 65))
weights3 = np.random.uniform(low=-0.001, high=0.001, size=(10, 65))

array_of_layers = [Layer(weights1), Layer(weights2), Layer(weights3)]
my_network = Network(array_of_layers)


IMAGE_MAGIC_NUMBER = b'\x00\x00\x08\x03'
LABEL_MAGIC_NUMBER = b'\x00\x00\x08\x01'

def read_mnist_file(path):
    contents = None
    with open(path, 'rb') as f:
        contents = f.read()
    if contents[:4] == IMAGE_MAGIC_NUMBER:
        return parse_images(contents)
    elif contents[:4] == LABEL_MAGIC_NUMBER:
        return parse_labels(contents)
    print(contents[:4])

def parse_images(c):  # normalized image array
    idx = 16
    out = []
    n = int(c[4:8].hex(), 16)
    img_size = 28 * 28
    for i in range(n):
        arr = [float(b) for b in c[idx:idx+img_size]]
        out.append(np.array(arr) / 255.0)
        idx += img_size
    return out

def parse_labels(c):  # labels as one-hot encodings
    idx = 8
    out = []
    n = int(c[4:8].hex(), 16)
    for i in range(n):
        oh = to_onehot(c[idx], 10)
        out.append(oh)
        idx += 1
    return out

def to_onehot(idx, out_of):
    z = np.zeros(out_of)
    z[idx] = 1.0
    return z

def column_vector_to_one_hot (cv):
    max = cv[0][0]
    index_of_max = 0
    for i in range(cv.size):
        if cv[i][0] > max:
            max = cv[i][0]
            index_of_max = i
    z = np.zeros([cv.size, 1])
    z[index_of_max] = [1]
    return z


training_images = read_mnist_file('./archive/train-images-idx3-ubyte')
training_labels = read_mnist_file('./archive/train-labels-idx1-ubyte')
testing_images = read_mnist_file('./archive/t10k-images-idx3-ubyte')
testing_labels = read_mnist_file('./archive/t10k-labels-idx1-ubyte')

epochs = 5
for epoch in range(epochs):
    for i in range(60000):
        my_network.train(training_images[i].reshape(-1, 1), training_labels[i].reshape(-1, 1))

num_correct = 0
for i in range(10000):
    if np.array_equal(column_vector_to_one_hot(my_network.forward_pass_network(testing_images[i].reshape(-1, 1))), testing_labels[i].reshape(-1, 1)):
        num_correct += 1

print(num_correct)
#my neural network gets >95% correct on MNIST!!!