import numpy as np
import matplotlib.pyplot as plt

# GLobal Parameters

signal_freq = 5.0   # Hz
duration = 2.0      # seconds
sampling_freq = 8   # Hz # samples per second
num_bits = 3       # bits -> 2**num_bits levels
min_signal = -1.0   # quantizer min
max_signal =  1.0   # quantizer max


# Signal Generator
def original_signal(t):
    """x(t) = sin(2*pi*f*t), works with scalars or numpy arrays."""
    return np.sin(2 * np.pi * signal_freq * t)

# Plotting the original signal
t_points = np.linspace(0, duration, 1000, endpoint=False)
cont_signal = original_signal(t_points)
plt.plot(t_points, cont_signal, label='Continuous Signal')


def sample_times(duration, sampling_freq, start=0.08):
# Sampling the original signal
    n_samples = int(sampling_freq * duration)
    t_sampled = np.linspace(start, start+duration, n_samples, endpoint=False)
    # t_sampled = np.linspace(0, duration, n_samples, endpoint=False) #
    return t_sampled

t_sampled = sample_times(duration, sampling_freq)
sampled_signal = original_signal(t_sampled)

def quantize_uniform(sampled_signal, num_bits, min_signal, max_signal):
    """
    Uniform mid-rise quantizer:
      n_levels = 2^n_bits levels; step Δ = (x_max - x_min) / (L - 1)
      q(x) = x_min + round((x - x_min)/Δ) * Δ
    Returns (qv, qs) = (quantized values, integer indices).
    """
    # Quantize the sampled signal
    n_levels = 2 ** num_bits
    # Map from [min,max] -> [0, n_levels-1], round to nearest integer
    qs = np.round((sampled_signal - min_signal) / (max_signal - min_signal) * (n_levels - 1))
    # Clip to ensure valid integer codebook
    qs = np.clip(qs, 0, n_levels - 1).astype(int)
    # Map codes back to quantized amplitudes in [min,max]
    qv = min_signal + qs * (max_signal - min_signal) / (n_levels - 1)
    #
    return qv, qs

qv, qs = quantize_uniform(sampled_signal, num_bits, min_signal, max_signal)


# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(9, 5))
# Continuous signal
plt.plot(t_points, cont_signal, label="Continuous signal", linewidth=2)

# Sampled points (stems). Shows values at sample times
plt.stem(t_sampled, sampled_signal, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Sampled signal")

# Quantized staircase at sample times (using step)
plt.step(t_sampled, qv, where='post', label=f'Quantized signal ({num_bits} bits)', color='C3', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Sampling at {sampling_freq} Hz and Quantization of a 5 Hz Sine")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()