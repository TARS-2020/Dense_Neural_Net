import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, Image
import AdamAI
import sys, os

def resource_path(filename):
    # Works both in Python and in exe
    base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(base_path, filename)

# --- Initialize network (same as yours) ---
model_path = resource_path("my_digit_vec_net.npz")
net = AdamAI.Network([12544, 2048, 1024, 512, 256, 256, 128, 128, 64, 10],
                    [0.0,   0.1,  0.1,  0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
AdamAI.load_network(net, model_path)

# --- Config ---
CANVAS_PIXELS = 112
SAVE_PIXELS = 112
SCALE = 4
BRUSH = 2
UPDATE_INTERVAL_MS = 100  # update predictions every 100 ms

class DrawPredict:
    def __init__(self, net, update_ms=UPDATE_INTERVAL_MS):
        self.net = net
        self.update_ms = update_ms

        self.root = tk.Tk()
        self.root.title("Draw for Prediction")

        # --- Drawing canvas (scaled) ---
        self.canvas = tk.Canvas(
            self.root, width=CANVAS_PIXELS*SCALE, height=CANVAS_PIXELS*SCALE, bg="black"
        )
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        # PIL image buffer (single channel)
        self.image = Image.new("L", (CANVAS_PIXELS, CANVAS_PIXELS), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Clear button
        tk.Button(self.root, text="Clear", command=self.clear).grid(row=3, column=0, pady=6)

        # --- Bar graph canvas ---
        self.bar_canvas_width = 420
        self.bar_canvas_height = 10 * 34 + 20
        self.bar_canvas = tk.Canvas(self.root, width=self.bar_canvas_width, height=self.bar_canvas_height, bg="white")
        self.bar_canvas.grid(row=0, column=1, padx=10, pady=10)

        # Layout constants for bars
        self.label_x = 10
        self.bar_x = 50
        self.max_bar_width = 320
        self.bar_height = 22
        self.bar_spacing = 12
        self.prob_x = self.bar_x + self.max_bar_width + 8

        # Create persistent rectangles and text items (one per digit)
        self.bar_rects = []
        self.label_texts = []
        self.prob_texts = []
        for i in range(10):
            y = 10 + i * (self.bar_height + self.bar_spacing)
            rect = self.bar_canvas.create_rectangle(self.bar_x, y, self.bar_x, y + self.bar_height, fill="skyblue", outline="")
            label = self.bar_canvas.create_text(self.label_x, y + self.bar_height/2, anchor="w", text=f"{i}", font=("Arial", 12, "bold"))
            prob_text = self.bar_canvas.create_text(self.prob_x, y + self.bar_height/2, anchor="w", text="0.00", font=("Arial", 11))
            self.bar_rects.append(rect)
            self.label_texts.append(label)
            self.prob_texts.append(prob_text)

        # Label for top prediction
        self.pred_var = tk.StringVar(value="Predicted: - (0.00)")
        tk.Label(self.root, textvariable=self.pred_var, font=("Arial", 14)).grid(row=1, column=1, pady=6)

        # Start live updates
        self.update_prediction()
        self.root.mainloop()

    def paint(self, event):
        x = event.x // SCALE
        y = event.y // SCALE
        for i in range(BRUSH):
            for j in range(BRUSH):
                xi = x + i
                yj = y + j
                if 0 <= xi < CANVAS_PIXELS and 0 <= yj < CANVAS_PIXELS:
                    color = 200
                    self.canvas.create_rectangle(
                        xi*SCALE, yj*SCALE, (xi+1)*SCALE, (yj+1)*SCALE,
                        fill="white", outline="white"
                    )
                    self.image.putpixel((xi, yj), color)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_PIXELS, CANVAS_PIXELS), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def update_prediction(self):
        # Resize to network input (same as your old code)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_resized = self.image.resize((SAVE_PIXELS, SAVE_PIXELS), resample)

        arr = np.array(img_resized).astype(np.float32) / 255.0
        output = self.net.network_forward_pass(arr.flatten())
        probs = np.array(output, dtype=np.float64)

        # Update bar rectangles and probability texts
        for i, p in enumerate(probs):
            p = float(np.clip(p, 0.0, 1.0))  # clamp
            y = 10 + i * (self.bar_height + self.bar_spacing)
            x1 = self.bar_x + p * self.max_bar_width
            self.bar_canvas.coords(self.bar_rects[i], self.bar_x, y, x1, y + self.bar_height)
            self.bar_canvas.itemconfig(self.prob_texts[i], text=f"{p:.2f}")

        # Update top prediction label
        pred_idx = int(np.argmax(probs))
        self.pred_var.set(f"Predicted: {pred_idx} ({probs[pred_idx]:.2f})")

        # schedule next update
        self.root.after(self.update_ms, self.update_prediction)


# Run
if __name__ == "__main__":
    DrawPredict(net)