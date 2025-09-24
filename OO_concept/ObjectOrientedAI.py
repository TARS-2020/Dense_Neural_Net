import numpy as np

def softmax(x):
        e_x = np.exp(x - np.max(x))  # numerical stability
        return e_x / e_x.sum()

class Neuron:
    def __init__(self, n_inputs, is_output = False):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = np.random.randn() * 0.1
        self.is_output = is_output
        self.last_output = None

    def activation(self, x):
        z = np.dot(x, self.weights) + self.bias
        if self.is_output:
            self.last_output = z   # softmax later at layer-level
            return z
        else:
            self.last_output = np.tanh(z)
            return self.last_output  # hidden -> tanh

    def activation_derivative(self, output):
        if self.is_output:
            # derivative for sigmoid
            return output * (1 - output)
        else:
            # derivative for tanh
            return 1 - output**2
    
class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron, is_output = False):
        self.neurons = [Neuron(n_inputs_per_neuron, is_output) for _ in range(n_neurons)]

    def forward(self, inputs):
        outputs = np.array([neuron.activation(inputs) for neuron in self.neurons])
        if self.neurons[0].is_output:  # apply softmax to whole layer
            outputs = softmax(outputs)
            for i, neuron in enumerate(self.neurons):
                neuron.last_output = outputs[i]
        return outputs


class Network:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of integers, e.g. [2, 4, 4, 1]
        means: 2 inputs -> 4 neurons -> 4 neurons -> 1 output
        """
        self.layers_output = []
        self.layers_input = []
        self.layers = []
        for i in range(1, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1], True))
            else:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1]))

    def forward(self, inputs):
        x = inputs
        self.layers_input = [x]       # input to the first layer
        self.layers_output = []        # store outputs of layers

        for layer in self.layers:
            layer_output = layer.forward(x)
            self.layers_output.append(layer_output)
            x = layer_output
            self.layers_input.append(x)  # input for the next layer

        return x

    
    def backward(self, input, target, lr = 0.1):

        prediction = self.forward(input)

        next_deltas = None

        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            layer_input = self.layers_input[layer_idx]

            deltas = []
            for neuron_idx, neuron in enumerate(layer.neurons):
                if layer_idx == len(self.layers) - 1:#Output Layer
                    delta = neuron.last_output - target[neuron_idx]
                else:#Hidden Layer
                    delta = sum(next_neuron.weights[neuron_idx] * next_deltas[i]
                                for i, next_neuron in enumerate(self.layers[layer_idx + 1].neurons))
                    delta *= neuron.activation_derivative(neuron.last_output)

                deltas.append(delta)
            
                #Update weights
                for w_idx in range(len(neuron.weights)):
                    neuron.weights[w_idx] -= lr * deltas[neuron_idx] * layer_input[w_idx]

                # Update bias
                neuron.bias -= lr * deltas[neuron_idx]

            next_deltas = deltas

        return prediction