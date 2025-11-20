import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Random weight matrix
# -----------------------------
np.random.seed(0)
W = np.random.uniform(-1.0, 1.0, size=(256, 256))

# -----------------------------
# 2. 4-bit uniform quantization
#    over [-1, 1]
# -----------------------------
num_bits = 4
num_levels = 2 ** num_bits            # 16
w_min, w_max = -1.0, 1.0
Delta = (w_max - w_min) / (num_levels - 1)  # step size

def quantize_uniform(w):
    w_clipped = np.clip(w, w_min, w_max)
    k = np.round((w_clipped - w_min) / Delta)
    return w_min + k * Delta

W_q = quantize_uniform(W)  # dequantized 4-bit weights

# -----------------------------
# 3. Cosine snapping:
#    w_cs = w + alpha * Delta * cos(theta * w)
# -----------------------------
alpha = 0.8      # snapping strength (play with this)
theta = 15.0     # frequency term (play with this)

W_cs = W + alpha * Delta * np.cos(theta * W)

# -----------------------------
# 4. 1D view for plotting
#    (to see function shapes)
# -----------------------------
def quantize_uniform_scalar(w):
    w_clipped = np.clip(w, w_min, w_max)
    k = np.round((w_clipped - w_min) / Delta)
    return w_min + k * Delta

w_line = np.linspace(-1.0, 1.0, 2000)
q_line = np.vectorize(quantize_uniform_scalar)(w_line)
w_cs_line = w_line + alpha * Delta * np.cos(theta * w_line)

# -----------------------------
# 5. Plots
# -----------------------------

# (A) Quantization vs cosine-snapping vs identity
plt.figure()
plt.plot(w_line, q_line, label="4-bit quantized (dequantized)")
plt.plot(w_line, w_cs_line, linestyle='--', label="Cosine snapping approx")
plt.plot(w_line, w_line, linestyle=':', label="Identity (w)")
plt.xlabel("Original weight w")
plt.ylabel("Output value")
plt.title("4-bit quantization vs cosine-snapping approximation")
plt.legend()
plt.grid(True)
plt.show()

# (B) Approximation error
error_line = w_cs_line - q_line
plt.figure()
plt.plot(w_line, error_line)
plt.xlabel("Original weight w")
plt.ylabel("Approx - Quantized")
plt.title("Approximation error: cosine snapping vs true 4-bit quantization")
plt.grid(True)
plt.show()