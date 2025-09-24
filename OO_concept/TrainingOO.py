import numpy as np
from ObjectOrientedAI import Network
from PIL import Image, ImageDraw
import tkinter as tk

def save_network(net, filename="network_state.npz"):
    """Save weights and biases of all layers"""
    params = {}
    for layer_idx, layer in enumerate(net.layers):
        for neuron_idx, neuron in enumerate(layer.neurons):
            params[f"layer{layer_idx}_neuron{neuron_idx}_w"] = neuron.weights
            params[f"layer{layer_idx}_neuron{neuron_idx}_b"] = neuron.bias
    np.savez(filename, **params)
    print(f"Network saved to {filename}")

def load_network(net, filename="network_state.npz"):
    """Load weights and biases into a network"""
    try:
        data = np.load(filename, allow_pickle=True)
        for layer_idx, layer in enumerate(net.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                neuron.weights = data[f"layer{layer_idx}_neuron{neuron_idx}_w"]
                neuron.bias = data[f"layer{layer_idx}_neuron{neuron_idx}_b"]
        print(f"Network loaded from {filename}")
        return
    except Exception as e:
        print("Could not load Network.", e)


# --- Load dataset ---
X = np.load("digits_images.npy")  # shape (N, 10000)
y = np.load("digits_labels.npy")  # shape (N,)

# Determine number of classes
num_classes = len(set(y))  # includes -1

# Map -1 to a positive index for one-hot encoding
label_map = {label: i for i, label in enumerate(sorted(set(y)))}
y_mapped = np.array([label_map[l] for l in y])

# One-hot encode labels
Y = np.zeros((len(y), len(label_map)))
for i, label_idx in enumerate(y_mapped):
    Y[i, label_idx] = 1

# --- Initialize network ---
input_size = X.shape[1]  # 10000
net = Network([input_size, 128, 64, len(label_map)])
load_network(net, "my_digit_net.npz")

# --- Training ---
epochs = 5
lr = 0.01

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        pred = net.backward(X[i], Y[i], lr)

        # Cross-entropy loss
        pred_clipped = np.clip(pred, 1e-10, 1-1e-10)
        loss = -np.sum(Y[i] * np.log(pred_clipped))
        total_loss += loss

    avg_loss = total_loss / len(X)
    if (epoch + 1) % 1 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

save_network(net, "my_digit_net.npz")

# --- Draw and predict ---
CANVAS_PIXELS = 200
SAVE_PIXELS = 100
SCALE = 2
BRUSH = 4

class DrawPredict:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw for Prediction")

        self.canvas = tk.Canvas(self.root, width=CANVAS_PIXELS*SCALE, height=CANVAS_PIXELS*SCALE, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (CANVAS_PIXELS, CANVAS_PIXELS), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.predict_btn = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=5)

        self.result_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.result_var, font=("Arial", 16)).pack(pady=5)

        tk.Button(self.root, text="Clear", command=self.clear).pack(pady=5)

        self.root.mainloop()

    def paint(self, event):
        x = event.x // SCALE
        y = event.y // SCALE
        for i in range(BRUSH):
            for j in range(BRUSH):
                xi = x + i
                yj = y + j
                if 0 <= xi < CANVAS_PIXELS and 0 <= yj < CANVAS_PIXELS:
                    self.canvas.create_rectangle(
                        xi*SCALE, yj*SCALE, (xi+1)*SCALE, (yj+1)*SCALE,
                        fill="black", outline="black"
                    )
                    self.image.putpixel((xi, yj), 0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_PIXELS, CANVAS_PIXELS), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Resize to SAVE_PIXELS
        img_resized = self.image.resize((SAVE_PIXELS, SAVE_PIXELS), Image.Resampling.LANCZOS)
        arr = np.array(img_resized).flatten() / 255.0
        output = net.forward(arr)
        predicted_idx = np.argmax(output)
        # Map back to original label
        inv_label_map = {v: k for k, v in label_map.items()}
        predicted_label = inv_label_map[predicted_idx]
        self.result_var.set(f"Predicted: {predicted_label} ({output[predicted_idx]:.2f})")
        print("Prediction probabilities:", output)

# Run the draw-and-predict window
DrawPredict()