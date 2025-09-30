# Camera System Analysis Report

## Contents

1. [Dark Frame Analysis](#part-1-dark-frame-analysis)
2. [Multi-Camera System Analysis](#part-2-multi-camera-system-analysis)
3. [Convolution Filtering of Live Video](#part-3-convolution-filtering-of-live-video-stream)

---

# Part 1: Dark Frame Analysis

## Experimental Setup

- Three dark frames captured with lens completely covered
- 100×100 pixel patch selected from center
- Metrics: **Mean (μ)** and **Std Dev (σ)**
- Histograms generated for each patch

## Raw Data

| File       | dtype   | Shape       | Value Range    | Patch μ  | Patch σ  |
| ---------- | ------- | ----------- | -------------- | -------- | -------- |
| dark_1.dng | float64 | (3024,4032) | –13106 → 13535 | –897.479 | 3728.599 |
| dark_2.dng | float64 | (3024,4032) | –13106 → 12666 | –839.741 | 3503.720 |
| dark_3.dng | float64 | (3024,4032) | –13106 → 14406 | –811.555 | 3732.325 |

Histograms saved in `./outputs/out_hist/`.

## Results & Discussion

- **Mean Values:** Around –800 to –900 → stable dark signal
- **Std Dev:** ~3500–3700 → noise from dark current + readout electronics
- **Histogram Consistency:** Similar across frames → noise pattern repeatable

### Practical Implications

- Even with no light, sensor produces measurable fluctuations
- Sets a lower bound for image quality in low-light scenes
- In practice, visible as graininess in smartphone night photos

---

# Part 2: Multi-Camera System Analysis

## Experimental Setup

- Raw images from main vs. ultrawide cameras
- Files:
  - Main: `sheep_main_lens.dng`
  - Ultrawide: `sheep_wide_lens.dng`
- Analysis: Field of view (FOV), raw values, noise, and SNR

---

## Step 1: Field of View (FOV)

| Camera    | Focal Length | Sensor Width | Horizontal FOV | Coverage vs. Main |
| --------- | ------------ | ------------ | -------------- | ----------------- |
| Main      | 5.1 mm       | 7 mm         | 68.92°         | 1.00×             |
| Ultrawide | 1.54 mm      | 4 mm         | 104.81°        | 1.52× wider       |

- **Difference:** Ultrawide is 35.9° wider (1.52× coverage)

---

## Step 2: Raw Image Properties

| Camera    | Black Level | Shape       | Value Range   |
| --------- | ----------- | ----------- | ------------- |
| Main      | 0.00        | (3024,4032) | 12842 → 65534 |
| Ultrawide | 8427.00     | (3024,4032) | –8427 → 57107 |

---

## Step 3: Noise Analysis

### Main Camera (FOV = 68.92°)

- Region: 73×103 pixels (7519 samples)
- μ = 23255.64
- σ = 4013.18
- SNR = 5.79

### Ultrawide Camera (FOV = 104.81°)

- Region: 66×108 pixels (7128 samples)
- μ = 32070.01
- σ = 4599.62
- SNR = 6.97

### Comparison

| Metric      | Main     | Ultrawide | Ratio (U/M) |
| ----------- | -------- | --------- | ----------- |
| Mean (μ)    | 23255.64 | 32070.01  | 1.38×       |
| Std Dev (σ) | 4013.18  | 4599.62   | 1.15×       |
| SNR (μ/σ)   | 5.79     | 6.97      | 1.20×       |

- Ultrawide has **more absolute noise**, but **better SNR** (cleaner relative signal).

---

## Step 4: Theoretical Background

- **FOV Equation:** `FOV = 2 × arctan(w / (2f))`
  - Shorter focal length → wider FOV
  - Longer focal length → narrower FOV
- **Noise Sources:**
  - Photon shot noise (∝ √N)
  - Read noise (electronics)
  - Dark current noise (thermal)
  - Quantization noise (ADC rounding)

**Key Insight:** Aperture (f-number = f/D) controls light collection. Larger aperture (smaller f-number) → more light → higher SNR.

---

## Step 5: Practical Applications

| Camera    | Advantages                          | Disadvantages                | Best For                    |
| --------- | ----------------------------------- | ---------------------------- | --------------------------- |
| Main      | Natural perspective, versatile lens | Narrower FOV, lower SNR      | Portraits, everyday shots   |
| Ultrawide | Expansive FOV, better SNR here      | Distortion, edge softness    | Landscapes, architecture    |
| Telephoto | Narrow FOV, subject isolation       | Worse low-light, specialized | Distant subjects, portraits |

---

# Part 3: Convolution Filtering of Live Video Stream

## Experimental Setup

- Webcam feed captured, converted to grayscale
- Manual convolution applied (no OpenCV built-ins)
- Filters tested: **Box, Gaussian, Sobel (H/V), Sharpen, Emboss**
- Displayed side-by-side with original

## Observations

| Filter     | Effect on Video                                 | Kernel Logic                                   |
| ---------- | ----------------------------------------------- | ---------------------------------------------- |
| Box (Blur) | Strong blur, smeared hand edges, loss of detail | Equal weights → averages neighbors             |
| Gaussian   | Smooth blur, reduced noise, details preserved   | Center weighted → natural smoothing            |
| Sobel H/V  | Highlights horizontal/vertical edges            | Positive vs. negative weights detect gradients |
| Sharpen    | Crisper, stronger edges, more noise visible     | Positive center + negative neighbors           |
| Emboss     | Relief-like effect, pseudo-3D shading           | Negative vs. positive diagonal weights         |

## Scene Experiments

- **Hand movement:** Sobel filters highlighted edges dynamically; blur smeared outlines
- **Well-lit object:** Sharpening filter made contours stand out; emboss gave artificial shadows
- **Textured surface:** Gaussian preserved patterns with reduced noise; emboss converted textures into relief patterns

---

## Final Summary

1. **Dark Frames:** Sensor produces measurable noise (σ ~ 3500) even with no light, limiting low-light performance.
2. **Multi-Camera:** Ultrawide captures 1.52× more scene and showed **better SNR (6.97 vs 5.79)** despite more absolute noise. Likely due to aperture, pixel size, or exposure differences.
3. **Filtering:** Different convolution kernels directly control how details are emphasized: blur smooths, Sobel extracts edges, sharpen boosts contrast, emboss adds depth-like shading.

**Key Takeaway:** Both optical hardware (focal length, aperture, sensor) and software processing (convolution kernels) demonstrate how physics and math shape the final image.
