import numpy as np
import matplotlib.pyplot as plt

# 1) Define 4-bit uniform quantizer on [-1, 1]
num_bits = 4
num_levels = 2 ** num_bits
w_min, w_max = -1.0, 1.0
Delta = (w_max - w_min) / (num_levels - 1)

def quantize_uniform(w):
    w_clipped = np.clip(w, w_min, w_max)
    k = np.round((w_clipped - w_min) / Delta)
    return w_min + k * Delta

# Sample dense grid of weights
w = np.linspace(-1.0, 1.0, 4000)
q = quantize_uniform(w)
r = q - w                     # residual to approximate

# 2) Build multi-cos features
M = 5                         # number of cosine terms
theta = 2 * np.pi / (2.0)     # period ~2 to match [-1,1]; you can tweak this

Phi = []
for m in range(1, M + 1):
    Phi.append(np.cos(m * theta * w))
Phi = np.stack(Phi, axis=1)   # shape (N, M)

# 3) Least-squares fit: r ≈ Delta * Phi @ alpha
#    => alpha = argmin ||Delta Phi alpha - r||^2
A = Delta * Phi
alpha, *_ = np.linalg.lstsq(A, r, rcond=None)

# 4) Build approximation and compare
r_hat = A @ alpha
w_hat = w + r_hat

mse_single = np.mean((quantize_uniform(w) - (w + Delta * np.cos(theta * w)))**2)
mse_multi  = np.mean((q - w_hat)**2)
print("MSE single-cos:", mse_single)
print("MSE multi-cos :", mse_multi)

# 5) Plots
plt.figure()
plt.plot(w, q, label="4-bit quantized")
plt.plot(w, w_hat, '--', label="multi-cos approx (M={})".format(M))
plt.plot(w, w, ':', label="identity w")
plt.xlabel("w")
plt.ylabel("output")
plt.title("4-bit quantization vs multi-cosine approximation")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(w, w_hat - q)
plt.xlabel("w")
plt.ylabel("approx - quantized")
plt.title("Error: multi-cos approx vs true 4-bit quantization")
plt.grid(True)
plt.show()