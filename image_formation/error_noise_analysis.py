# noise_sampling_quantization.py
import numpy as np
import matplotlib.pyplot as plt
from sampling_quantization import original_signal, sample_times, quantize_uniform

# Global Parameters
signal_freq = 5.0    # Hz
duration = 2.0       # seconds
sampling_freq = 8   # sampling frequency in Hz
num_bits = 3         # bits -> 2**num_bits levels
min_signal = -1.0    # quantizer min (match to signal range)
max_signal =  1.0    # quantizer max

# Noise (Gaussian) parameters
mean = 0.0
std_dev = 0.10       # relative noise level (scaled by signal magnitude)
# For Reproducibility, setting seed
np.random.seed(0)


def add_Gaussian_noise(signal, mean, std):
    """
    Additive Gaussian noise scaled by the signal magnitude to
    ensure noise is proportional to the signal magnitude:
    noise ~ N(mean, (std * (max - min))^2)
    """
    mag = np.max(signal) - np.min(signal)  # dynamic range of the (sampled) signal
    noise = np.random.normal(mean, std * mag, size=len(signal))
    return signal + noise, noise

# x = signal, n_bits = n_levels, x_min = min_signal, x_max=max_signal, L = n_levels


def mse(a, b):
    """Mean squared error."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.mean((a - b) ** 2)

def rmse(a, b):
    return np.sqrt(mse(a, b))

def psnr(ref, test, peak=None):
    """
    PSNR = 10 * log10( peak^2 / MSE(ref,test) )
    peak defaults to the absolute max magnitude of the reference.
    """
    err = mse(ref, test)
    if err == 0:
        return np.inf
    if peak is None:
        peak = np.max(np.abs(ref))
    return 10.0 * np.log10((peak ** 2) / err)

# -----------------------------
# Plotting helper
# -----------------------------
def plot_all(t_dense, x_dense, t_s, x_s_noisy, x_q, title_suffix=""):
    plt.figure(figsize=(10, 5))
    # Continuous reference
    plt.plot(t_dense, x_dense, label="Continuous signal", linewidth=2)
    # Sampled noisy points
    plt.plot(t_s, x_s_noisy, "o", label="Sampled + noise", alpha=0.9)
    # Quantized staircase
    plt.step(t_s, x_q, where="post", label=f"Quantized ({num_bits} bits)", color="C3", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    ttl = "Sampling, Noise, and Quantization of a 5 Hz Sine"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main flow
# -----------------------------
def main():
    # Dense time for continuous reference
    t_dense = np.linspace(0, duration, 1000, endpoint=False)
    x_dense = original_signal(t_dense)

    # Sample (avoid exact zero-crossing alignment with a tiny phase offset if desired)
    t_s = sample_times(duration, sampling_freq, start=0.0)
    x_s_clean = original_signal(t_s)

    # Add Gaussian noise to the *sampled* signal
    x_s_noisy, noise_vec = add_Gaussian_noise(x_s_clean, mean, std_dev)

    # Quantize the noisy samples
    x_q, _ = quantize_uniform(x_s_noisy, num_bits, min_signal, max_signal)

    # --- Metrics ---
    # Compare clean samples vs noisy samples
    mse_noisy = mse(x_s_clean, x_s_noisy)
    rmse_noisy = np.sqrt(mse_noisy)
    psnr_noisy = psnr(x_s_clean, x_s_noisy, peak=np.max(np.abs([min_signal, max_signal])))

    # Compare clean samples vs quantized (noisy) samples
    mse_quant = mse(x_s_clean, x_q)
    rmse_quant = np.sqrt(mse_quant)
    psnr_quant = psnr(x_s_clean, x_q, peak=np.max(np.abs([min_signal, max_signal])))

    print(f"Sampling: fs={sampling_freq} Hz, duration={duration}s, samples={len(t_s)}")
    print(f"Quantization: bits={num_bits}, levels={2**num_bits}, range=[{min_signal}, {max_signal}]")
    print(f"Noise: mean={mean}, std_dev={std_dev} (scaled by signal magnitude)")

    print("\n--- Error vs CLEAN SAMPLES ---")
    print(f" Noisy samples:    MSE={mse_noisy:.6f}  RMSE={rmse_noisy:.6f}  PSNR={psnr_noisy:.2f} dB")
    print(f" Quantized (noisy): MSE={mse_quant:.6f}  RMSE={rmse_quant:.6f}  PSNR={psnr_quant:.2f} dB")

    # Plot
    plot_all(t_dense, x_dense, t_s, x_s_noisy, x_q,
             title_suffix=f"fs={sampling_freq} Hz, {num_bits} bits, noise σ={std_dev}")

if __name__ == "__main__":
    main()
