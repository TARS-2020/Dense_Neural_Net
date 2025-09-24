import struct
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import AdamAI
import DataProcessing
import multiprocessing as mp
import time

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

def evaluate_network(network, X, y, batch_size=2000, num_classes=10):
    correct = 0
    total_loss = 0
    num_samples = len(X)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Forward pass once
        predictions = network.network_batch_forward_pass(X_batch, training=False)
        pred_clipped = np.clip(predictions, 1e-10, 1 - 1e-10)

        # Accuracy
        predicted_labels = np.argmax(pred_clipped, axis=1)
        correct += np.sum(predicted_labels == y_batch)

        # One-hot encode batch for loss
        targets = np.zeros((len(y_batch), num_classes))
        targets[np.arange(len(y_batch)), y_batch] = 1

        # Cross-entropy loss
        batch_loss = -np.sum(targets * np.log(pred_clipped)) / len(X_batch)
        total_loss += batch_loss * len(X_batch)

    accuracy = correct / num_samples
    avg_loss = total_loss / num_samples
    return accuracy, avg_loss

def worker_augmenter(task_q, result_q, X_original, y_original, **aug_params):
    """
    Worker that repeatedly generates augmented datasets and pushes them to result_q.
    """
    while True:
        num_samples = task_q.get()
        if num_samples is None:
            break  # shutdown signal

        # call your existing augmentation function
        X_aug, y_aug = DataProcessing.create_augmented_training_set(
            X_original, y_original,
            num_samples=num_samples,
            **aug_params
        )

        # immediately cast to float16 to save memory
        X_aug = X_aug.astype(np.float16)

        # put into queue (blocks if 2 datasets already waiting)
        result_q.put((X_aug, y_aug))

def start_pipeline(aug_params, max_prefetch=2):
    """
    Create the queues and start the worker.
    """
    X_original = load_images("data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte")
    y_original = load_labels("data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    
    task_q = mp.Queue()
    result_q = mp.Queue(maxsize=max_prefetch)

    worker = mp.Process(
        target=worker_augmenter,
        args=(task_q, result_q, X_original, y_original),
        kwargs=aug_params
    )
    worker.daemon = True
    worker.start()
    return task_q, result_q, worker

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # --- Initialize network ---
    net = AdamAI.Network([12544, 2048, 1024, 512, 256, 256, 128, 128, 64, 10],
                         [0.0,   0.1,  0.1,  0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    AdamAI.load_network(net, "my_digit_vec_net.npz")

    # --- Training parameters ---

    epochs = 0
    lr = 0.00025
    batch_size = 2500
    epoch_size = 20000
    validate = True
    label_map = {i:i for i in range(10)}

    if epochs != 0:
        #Load data
        try:
            aug_params = {
                "canvas_size": (112, 112),
                "min_scale": 0.5,
                "max_scale": 1.5,
                "max_angle": 10,
                "noise": 0.01,
                "need_noise": True
            }

            task_q, result_q, worker = start_pipeline(aug_params)

            task_q.put(epoch_size)  # first dataset
            task_q.put(epoch_size)  # second dataset (prefilled)
        except:
            raise

        for epoch in range(epochs):

            # if epoch % 10 == 0:
            #     lr = lr * 0.95

            X, y = result_q.get()

            X = X.reshape(X.shape[0], -1)

            task_q.put(epoch_size)

            # Map -1 to a positive index for one-hot encoding
            y_mapped = np.array([label_map[l] for l in y])

            # One-hot encode labels
            Y = np.zeros((len(y), len(label_map)))
            for i, label_idx in enumerate(y_mapped):
                Y[i, label_idx] = 1

            num_samples = X.shape[0]

            # Shuffle dataset indices
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            total_loss = 0
            num_batches = 0
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                X_batch = X[batch_indices]         # (batch_size, input_dim)
                Y_batch = Y[batch_indices]         # (batch_size, n_classes)

                # Forward + backward in batch mode
                predictions = net.network_batch_back_prop(X_batch, Y_batch, lr)

                # Cross-entropy loss
                pred_clipped = np.clip(predictions, 1e-10, 1 - 1e-10)
                batch_loss = -np.sum(Y_batch * np.log(pred_clipped)) / len(X_batch)

                total_loss += batch_loss
                num_batches += 1

                #print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {batch_loss:.4f}")

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

            #Backup
            if 0.5 < avg_loss < 1:
                AdamAI.save_network(net, "my_digit_vec_net_under1.npz")
            elif 0.4 < avg_loss < 0.5:
                AdamAI.save_network(net, "my_digit_vec_net_under05.npz")
            elif 0.3 < avg_loss < 0.4:
                AdamAI.save_network(net, "my_digit_vec_net_under04.npz")
            elif 0.2 < avg_loss < 0.3:
                AdamAI.save_network(net, "my_digit_vec_net_under03.npz")
            elif 0.1 < avg_loss < 0.2:
                AdamAI.save_network(net, "my_digit_vec_net_under02.npz")
            elif avg_loss < 0.1:
                AdamAI.save_network(net, "my_digit_vec_net_under01.npz")
                AdamAI.save_network(net, "my_digit_vec_net.npz")
                break

            # Save network after each epoch
            AdamAI.save_network(net, "my_digit_vec_net.npz")
            del X, y

            if validate and epoch % 10 == 0: #and epoch != 0:
                try:
                    X = np.load("mnist_augmented_batches/images_batch_11.npz")['arr_0']
                    X = X.reshape(X.shape[0], -1)
                    y = np.load("mnist_augmented_batches/labels_batch_11.npz")['arr_0']
                except Exception as e:
                    raise RuntimeError("Failed to load validation data") from e

                val_acc, val_loss = evaluate_network(net, X, y, batch_size)
                print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc*100:.2f}%")
                del X, y
            
        task_q.put(None)
        worker.join()