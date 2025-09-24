import numpy as np

def save_network(network, filename):
    save_dict = {"t": network.t}
    for idx, layer in enumerate(network.network_layers):
        save_dict[f"W{idx}"] = layer.weights
        save_dict[f"b{idx}"] = layer.biases
        save_dict[f"m_w{idx}"] = layer.m_w
        save_dict[f"v_w{idx}"] = layer.v_w
        save_dict[f"m_b{idx}"] = layer.m_b
        save_dict[f"v_b{idx}"] = layer.v_b
    np.savez(filename, **save_dict)

def load_network(network, filename):
    try:
        data = np.load(filename)
        network.t = int(data["t"])
        for idx, layer in enumerate(network.network_layers):
            layer.weights = data[f"W{idx}"]
            layer.biases  = data[f"b{idx}"]
            layer.m_w     = data[f"m_w{idx}"]
            layer.v_w     = data[f"v_w{idx}"]
            layer.m_b     = data[f"m_b{idx}"]
            layer.v_b     = data[f"v_b{idx}"]
        print(f"Network loaded from {filename}")
    except Exception as e:
        print("Could not load Network.", e)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Layer:
    def __init__(self, layer_inputs, layer_neuron_ammount, is_output = False, dropout = 0.0):
        self.weights = np.random.randn(layer_neuron_ammount, layer_inputs) * np.sqrt(2 / layer_inputs)
        self.biases = np.random.randn(layer_neuron_ammount) * 0.1 
        self.is_output = is_output
        self.last_layer_input = None
        self.last_layer_output = None

        #dropout variables
        self.dropout = dropout  # fraction of neurons to drop
        self.dropout_mask = None  # to store the mask during forward pass

        # --- Adam variables ---
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)

    def layer_activation(self, x):
        if self.is_output:
            return x
        else:
            return np.maximum(0, x)  # ReLU
            #return np.tanh(x)
        
    def layer_activation_derivative(self, output):
        if self.is_output:
            print("Output layer derivative handled in backward with softmax+CE")
            return None
        else:
            return (output > 0).astype(float) # ReLU derivative: 1 if output > 0 else 0
            #return 1 - output**2 #tanh derivative
        
    def layer_forward_pass(self, x, training = False):
        self.last_layer_input = x
        z = self.weights @ x + self.biases
        a = self.layer_activation(z)

        if training and self.dropout > 0.0 and not self.is_output:
            # create dropout mask
            self.dropout_mask = (np.random.rand(*a.shape) >= self.dropout).astype(float)
            # scale activations to keep expected value the same
            a *= self.dropout_mask / (1.0 - self.dropout)
        else:
            self.dropout_mask = None

        self.last_layer_output = a
        return a
    
    def layer_batch_forward_pass(self, X, training = False):
        self.last_layer_input = X
        Z = X @ self.weights.T + self.biases
        A = self.layer_activation(Z)

        if training and self.dropout > 0.0 and not self.is_output:
            self.dropout_mask = (np.random.rand(*A.shape) >= self.dropout).astype(float)
            A *= self.dropout_mask / (1.0 - self.dropout)
        else:
            self.dropout_mask = None

        self.last_layer_output = A
        return A

        
class Network:
    def __init__(self, layers_array, dropout_rates=None):
        self.network_layers = []
        self.t = 0  # global step counter for Adam

        if dropout_rates is None:
            dropout_rates = [0.0] * (len(layers_array)-1)

        for i in range(1, len(layers_array)):
            if i == len(layers_array) - 1:
                is_output = True
            else:
                is_output = False
            
            self.network_layers.append(Layer(layers_array[i-1], layers_array[i], is_output))

    def network_forward_pass(self, x, training = False):
        for layer in self.network_layers:
            x = layer.layer_forward_pass(x, training)

        #apply softmax on x that was provided by the last layer in the for loop above
        x = softmax(x)
        self.network_layers[-1].last_layer_output = x
        return x
    
    def network_back_prop(self, input, target, lr):
        prediction = self.network_forward_pass(input, True)
        next_delta = None

        for i in reversed(range(len(self.network_layers))):
            layer = self.network_layers[i]

            if i == len(self.network_layers) - 1:
                delta = prediction - target
            else:
                delta = (self.network_layers[i + 1].weights.T @ next_delta) * layer.layer_activation_derivative(layer.last_layer_output)
            
            # Apply dropout mask if this is a hidden layer with dropout
            if layer.dropout_mask is not None:
                delta *= layer.dropout_mask / (1.0 - layer.dropou)

            if i > 0:
                layer.weights -= lr * np.outer(delta, self.network_layers[i-1].last_layer_output)
            else:
                layer.weights -= lr * np.outer(delta, input)
            layer.biases -= lr * delta

            next_delta = delta

        return prediction
    
    def network_batch_forward_pass(self, X, training = True):
        for layer in self.network_layers:
            X = layer.layer_batch_forward_pass(X, training)

        # final softmax, row-wise
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        X = exps / np.sum(exps, axis=1, keepdims=True)
        self.network_layers[-1].last_layer_output = X
        return X
    
    def network_batch_back_prop(self, X_input, y_target, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        prediction = self.network_batch_forward_pass(X_input, True)
        next_delta = None
        batch_size = X_input.shape[0]

        for i in reversed(range(len(self.network_layers))):
            layer = self.network_layers[i]

            # Compute delta
            if i == len(self.network_layers) - 1:
                delta = prediction - y_target
            else:
                next_layer = self.network_layers[i + 1]
                delta = (next_delta @ next_layer.weights) * layer.layer_activation_derivative(layer.last_layer_output)

            # Apply dropout mask if this is a hidden layer with dropout
            if layer.dropout_mask is not None:
                delta *= layer.dropout_mask / (1.0 - layer.dropout)

            # Compute gradients
            grad_w = (delta.T @ layer.last_layer_input) / batch_size
            grad_b = delta.mean(axis=0)

            # --- Adam moving averages ---
            layer.m_w = beta1 * layer.m_w + (1 - beta1) * grad_w
            layer.v_w = beta2 * layer.v_w + (1 - beta2) * (grad_w ** 2)
            layer.m_b = beta1 * layer.m_b + (1 - beta1) * grad_b
            layer.v_b = beta2 * layer.v_b + (1 - beta2) * (grad_b ** 2)

            # --- Bias correction ---
            m_w_hat = layer.m_w / (1 - beta1 ** self.t)
            v_w_hat = layer.v_w / (1 - beta2 ** self.t)
            m_b_hat = layer.m_b / (1 - beta1 ** self.t)
            v_b_hat = layer.v_b / (1 - beta2 ** self.t)

            # --- Update weights and biases ---
            layer.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            layer.biases  -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            next_delta = delta

        return prediction