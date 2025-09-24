import struct
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import os

# -----------------------------
# MNIST Loading
# -----------------------------
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        return data.astype(np.float32) / 255.0
    
def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def mnist_to_pil_images(X):
    images = []
    for i in range(X.shape[0]):
        img_array = (X[i].reshape(28, 28) * 255).astype(np.uint8)
        images.append(Image.fromarray(img_array))
    return images

def create_augmented_training_set(X_original, y_original,
                                   num_samples=5000,
                                   canvas_size=(112,112),
                                   min_scale=0.5, max_scale=2.0, max_angle=45,
                                   output_prefix="train",
                                   noise = 0.02, need_noise = False):
    """
    Take random images from the MNIST dataset (excluding last 5000 for validation),
    apply augmentation, and save as .npz files.

    Args:
        X_original (np.ndarray): Original MNIST images (N, 784) normalized [0,1].
        y_original (np.ndarray): Labels (N,).
        num_samples (int): Number of random samples to pick.
        canvas_size (tuple): Output canvas size.
        min_scale, max_scale, max_angle: Augmentation params.
        output_prefix (str): File prefix for saving .npz files.
    """
    # Exclude last 5000 images (validation set)
    train_X = X_original[:-5000]
    train_y = y_original[:-5000]

    # Pick random 5000 images
    indices = np.random.choice(len(train_X), size=num_samples, replace=False)
    X_sel = train_X[indices]
    y_sel = train_y[indices]

    # Convert to PIL and augment
    X_aug, y_aug = [], []
    for img_array, label in zip(X_sel, y_sel):
        img_pil = Image.fromarray((img_array.reshape(28,28) * 255).astype(np.uint8))
        aug_img = augment_mnist_image(
            img_pil, canvas_size=canvas_size,
            min_scale=min_scale, max_scale=max_scale, max_angle=max_angle
        )
        X_aug.append(np.array(aug_img, dtype=np.float32) / 255.0)
        y_aug.append(label)

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    if need_noise:
        mask = np.random.rand(X_aug.shape[0]) < 0.5 # 50% chance for noise
        for i in range(0, X_aug.shape[0], 1000):  # process in chunks
            batch_mask = mask[i:i+1000]
            if np.any(batch_mask):  # only add noise where mask is True
                noise_arr = np.random.normal(0, noise, X_aug[i:i+1000].shape).astype(np.float32)
                X_aug[i:i+1000][batch_mask] += noise_arr[batch_mask]
        X_aug = np.clip(X_aug, 0, 1).astype(np.float16)



    # Save
    #np.savez_compressed(f"{output_prefix}_augmented_images.npz", X_aug)
    #np.savez_compressed(f"{output_prefix}_augmented_labels.npz", y_aug)

    #print(f"Saved {num_samples} augmented images to {output_prefix}_augmented_images.npz")
    #print(f"Saved labels to {output_prefix}_augmented_labels.npz")
    return X_aug, y_aug


# -----------------------------
# Augmentation
# -----------------------------
def augment_mnist_image(img, canvas_size=(112,112), min_scale=0.5, max_scale=2.0, max_angle=45):
    canvas_w, canvas_h = canvas_size

    if random.random() < 0.05:
        canvas = Image.new("L", (canvas_w, canvas_h), 0)  # black background
        x = (canvas_w - img.width) // 2
        y = (canvas_h - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas

    #Random scaling
    scale = random.uniform(min_scale, max_scale)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

    #Random rotation
    angle = random.uniform(-max_angle, max_angle)
    img_rotated = img_scaled.rotate(angle, expand=True, fillcolor=0)  # black fill

    #Ensure image fits canvas at random position
    img_w, img_h = img_rotated.size
    if img_w > canvas_w or img_h > canvas_h:
        scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        img_rotated = img_rotated.resize((int(img_w*scale_factor), int(img_h*scale_factor)), Image.BILINEAR)
        img_w, img_h = img_rotated.size

    max_x = canvas_w - img_w
    max_y = canvas_h - img_h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    #Paste onto black canvas
    canvas = Image.new('L', (canvas_w, canvas_h), 0)
    canvas.paste(img_rotated, (x, y))

    return canvas

# -----------------------------
# MNIST Augmented Viewer GUI
# -----------------------------

class MNISTViewer:
    def __init__(self, master, images, labels):
        self.master = master
        self.images = images
        self.labels = labels
        self.index = 0
        
        master.title("MNIST Augmented Viewer")
        
        # Image display label
        self.img_label = tk.Label(master)
        self.img_label.pack(padx=10, pady=10)
        
        # Digit label
        self.digit_label = tk.Label(master, text="", font=("Arial", 20))
        self.digit_label.pack(padx=5, pady=5)
        
        # Buttons frame
        frame = tk.Frame(master)
        frame.pack(pady=5)
        self.prev_button = tk.Button(frame, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = tk.Button(frame, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Show first image
        self.show_image()
        
    def show_image(self):
        # Convert normalized [0,1] image to 8-bit
        img_array = (self.images[self.index] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Optional: upscale for better viewing
        img = img.resize((224, 224), Image.NEAREST)
        
        # Display image in label
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)
        
        # Display corresponding label
        self.digit_label.config(text=f"Label: {self.labels[self.index]}")
        
    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_image()
            
    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

#X_batch, y_batch = create_augmented_training_set(X, y, num_samples=5000)

# Load one batch file
#X_batch = np.load("mnist_augmented_batches/images_batch_0.npz")['arr_0']
#y_batch = np.load("mnist_augmented_batches/labels_batch_0.npz")['arr_0']

# Launch GUI with this batch
#root = tk.Tk()
#viewer = MNISTViewer(root, X_batch, y_batch)
#root.mainloop()