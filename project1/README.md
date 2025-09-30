# Camera System Analysis Report

## Part 1: Dark Frame Analysis & Part 2: Multi-Camera Comparison

---

# Part 1: Dark Frame Analysis

## Experimental Setup

We captured three dark frames with the camera lens completely covered to measure sensor noise characteristics. From each image, we selected a 100×100 pixel patch from the center and measured the average brightness (mean) and the variation in pixel values (standard deviation).

## Raw Data

### File: dark_1.dng

- **dtype:** float64
- **shape:** (3024, 4032)
- **value range:** min: -13106.000, max: 13535.000
- **patch statistics (100×100 pixels):**
  - mean: -897.479
  - std: 3728.599
- **output:** histogram saved to `./outputs/out_hist/dark_1.dng.png`

### File: dark_2.dng

- **dtype:** float64
- **shape:** (3024, 4032)
- **value range:** min: -13106.000, max: 12666.000
- **patch statistics (100×100 pixels):**
  - mean: -839.741
  - std: 3503.720
- **output:** histogram saved to `./outputs/out_hist/dark_2.dng.png`

### File: dark_3.dng

- **dtype:** float64
- **shape:** (3024, 4032)
- **value range:** min: -13106.000, max: 14406.000
- **patch statistics (100×100 pixels):**
  - mean: -811.555
  - std: 3732.325
- **output:** histogram saved to `./outputs/out_hist/dark_3.dng.png`

## Results & Discussion

### Summary Statistics

**Table 1. Dark frame measurement results**

| Frame      | Mean (μ) | Standard Deviation (σ) |
| ---------- | -------- | ---------------------- |
| dark_1.dng | -897.48  | 3728.60                |
| dark_2.dng | -839.74  | 3503.72                |
| dark_3.dng | -811.55  | 3732.33                |

### Analysis

**Mean Values:**
The mean values are all close to –800 to –900. This shows that the overall signal level in the dark is stable, with only small shifts between frames.

**Standard Deviation:**
The standard deviation values are around 3500–3700. This number tells us how much the pixel values vary around the mean. Since no light was entering the camera, this variation comes from the sensor itself — mainly dark current noise (caused by thermal electrons inside the sensor) and readout noise (caused by the electronics when converting the signal to digital values).

**Histogram Consistency:**
The three histograms of pixel values all look very similar, with a spread around zero. This shows that the noise pattern is consistent and repeatable.

### Practical Implications

- Even when no light enters the camera, the sensor still produces a measurable signal that fluctuates.
- This noise sets a lower limit on how clean an image can be, especially in low-light situations.
- In everyday smartphone photos, this type of noise is most visible in dark scenes, where the true signal is small compared to the noise.

---

# Part 2: Multi-Camera System Analysis

## Experimental Setup

Comparison of Main Camera vs. Ultrawide Camera using raw sensor data (DNG format) without post-processing.

**Images analyzed:**

- Main camera: `./images/sheep_main_lens.dng`
- Ultrawide camera: `./images/sheep_wide_lens.dng`

## Step 1: Field of View (FOV) Calculation

### Camera Specifications and Results

**Main Camera:**

- **Focal Length:** 5.1 mm
- **Sensor Width:** 7 mm
- **Horizontal FOV:** 68.92°

**Ultrawide Camera:**

- **Focal Length:** 1.54 mm
- **Sensor Width:** 4 mm
- **Horizontal FOV:** 104.81°

**FOV Difference:** 35.89° (Ultrawide is **1.52×** wider)

---

## Step 2: Loading Raw Images

### Main Camera (sheep_main_lens.dng)

- **Black level:** 0.00
- **Image shape:** (3024, 4032)
- **Value range:** [12842.00, 65534.00]

### Ultrawide Camera (sheep_wide_lens.dng)

- **Black level:** 8427.00
- **Image shape:** (3024, 4032)
- **Value range:** [-8427.00, 57107.00]

---

## Step 3: Noise Analysis - Main Camera

### Main Camera Results

**Field of View:** 68.92°

**Noise Analysis (Raw Sensor Data)**

- **Region size:** 73×103 pixels (7519 total)

| Metric                | Value     |
| --------------------- | --------- |
| Mean (μ)              | 23255.64  |
| Std Dev (σ)           | 4013.1771 |
| Signal-to-Noise (SNR) | 5.79      |

---

## Step 4: Noise Analysis - Ultrawide Camera

### Ultrawide Camera Results

**Field of View:** 104.81°

**Noise Analysis (Raw Sensor Data)**

- **Region size:** 66×108 pixels (7128 total)

| Metric                | Value     |
| --------------------- | --------- |
| Mean (μ)              | 32070.01  |
| Std Dev (σ)           | 4599.6192 |
| Signal-to-Noise (SNR) | 6.97      |

---

## Step 5: Comparison Summary

### Optical Properties

| Camera    | FOV     | Scene Coverage   |
| --------- | ------- | ---------------- |
| Main      | 68.92°  | 1.00× (baseline) |
| Ultrawide | 104.81° | 1.52× wider      |

### Noise Comparison (Raw Sensor Data)

**Main Camera:**

- Mean (μ): 23255.64
- Std Dev (σ): 4013.1771
- SNR (μ/σ): 5.79

**Ultrawide Camera:**

- Mean (μ): 32070.01
- Std Dev (σ): 4599.6192
- SNR (μ/σ): 6.97

### Key Metrics

| Comparison Metric                    | Value | Interpretation                               |
| ------------------------------------ | ----- | -------------------------------------------- |
| Noise Ratio (σ_ultrawide / σ_main)   | 1.146 | Ultrawide has **1.15×** more noise than Main |
| SNR Ratio (SNR_ultrawide / SNR_main) | 1.203 | Ultrawide has **1.20×** better SNR than Main |

---

# Detailed Analysis Report

## 1. Field of View (FOV) Analysis

### Calculated FOV Values

From the measurements using **FOV = 2 × arctan(w / (2f))**:

**Main Camera:**

- Focal Length (f): 5.1 mm
- Sensor Width (w): 7 mm
- **Horizontal FOV: 68.92°**

**Ultrawide Camera:**

- Focal Length (f): 1.54 mm
- Sensor Width (w): 4 mm
- **Horizontal FOV: 104.81°**

The ultrawide camera captures approximately **1.52× more of the scene** horizontally compared to the main camera.

### Relationship Between Focal Length and FOV

The equation **FOV = 2 × arctan(w / (2f))** demonstrates an **inverse relationship** between focal length and field of view:

**Mathematical relationship:**

- As focal length (f) increases → FOV decreases
- As focal length (f) decreases → FOV increases

**Physical interpretation:**

The focal length determines where parallel light rays converge after passing through the lens. According to the thin lens formula **(1/f = 1/o + 1/i)**, for objects at infinity (o → ∞), the image forms at the focal point (i = f).

**Ultrawide Camera (Short focal length, f = 1.54mm):**

- The sensor is positioned closer to the lens
- Light rays from wide angles can still converge onto the sensor
- The angular acceptance cone is large
- **Result:** Wide field of view captures more of the scene

**Main Camera (Longer focal length, f = 5.1mm):**

- The sensor is positioned farther from the lens than the ultrawide
- Only light rays traveling at moderate angles to the optical axis reach the sensor
- The angular acceptance cone is narrower than ultrawide
- **Result:** Narrower field of view compared to ultrawide, but still moderate

**Note:** A true telephoto lens (e.g., 77mm) would have an even longer focal length, positioning the sensor much farther from the lens and accepting only near-parallel light rays, creating a very narrow "zoomed-in" view.

### Comparison to Telephoto Perspective

While this analysis compares main and ultrawide cameras, it's instructive to consider how a **telephoto lens** would differ. A telephoto lens (with even longer focal length, e.g., 77mm) would have:

- An even narrower FOV than the main camera (perhaps 15° or less)
- A stronger "zoomed-in" effect that compresses perspective
- The ability to isolate distant subjects

The telephoto's narrow angular acceptance excludes peripheral elements, making distant objects occupy a larger portion of the frame and creating the perception of being "closer" to the subject.

---

## 2. Noise Statistics Analysis

### Measured Noise Data

From the raw sensor analysis (Bayer pattern, no post-processing):

**Main Camera:**

- Mean (μ): 23255.64
- Standard Deviation (σ): 4013.1771
- Signal-to-Noise Ratio (SNR = μ/σ): 5.79

**Ultrawide Camera:**

- Mean (μ): 32070.01
- Standard Deviation (σ): 4599.6192
- Signal-to-Noise Ratio (SNR = μ/σ): 6.97

**Noise Ratio (σ_ultrawide / σ_main):** 1.146

### Theoretical Background: Aperture and Light Collection

The assignment mentions telephoto lenses typically having smaller apertures. While we're comparing main and ultrawide cameras here, the principle applies: aperture size affects light collection.

From the repository assignment, aperture is characterized by the f-number:

```
f-number = f / D
```

where f = focal length and D = aperture diameter.

**Key relationship:** A smaller f-number means a larger aperture, which collects more light.

**Light collection and aperture area:**

The amount of light collected is proportional to the aperture area:

```
Light ∝ π(D/2)² = π(f/(2·f-number))²
```

For the same f-number, a lens with longer focal length has a physically larger aperture diameter, but the f-number normalizes this effect.

### Noise Sources in Digital Sensors

From the repository assignment, we studied that sensor noise has multiple sources:

1. **Photon shot noise:** σ_shot ∝ √N, where N is the number of photons

   - Fundamental quantum uncertainty in light detection
   - Higher signal (more photons) → better SNR

2. **Read noise:** Fixed electronic noise from sensor readout

   - Independent of signal level
   - Dominates in low-light conditions

3. **Dark current noise:** Thermal electrons generated in the sensor
   - Temperature-dependent

**Signal-to-Noise Ratio:**

```
SNR = Signal / Total_Noise = μ / σ
```

Higher SNR indicates better image quality with less visible noise.

### Analysis of Results

**Expected behavior:**

Ultrawide cameras often have smaller apertures (higher f-numbers) compared to the main camera. If this is the case, the ultrawide would:

- Collect less light per unit time
- Have lower signal levels (smaller μ)
- Have worse SNR, especially in low-light conditions
- Show higher relative noise (larger σ/μ ratio)

Alternatively, some ultrawide cameras compensate with larger apertures or larger pixels to maintain competitive performance.

**Verification with data:**

**Scenario B - Ultrawide has better SNR:**

The data **refutes** the simple theoretical prediction. Despite expectations, the ultrawide camera shows:

- **38% higher mean signal** (32070 vs 23256)
- **15% higher noise standard deviation** (4600 vs 4013)
- **1.20× better SNR** compared to the main camera (6.97 vs 5.79)

**Possible explanations:**

1. **Larger aperture in ultrawide:** The ultrawide may actually have a larger relative aperture (smaller f-number), compensating for the shorter focal length. This would allow more light collection despite the wider FOV.

2. **Pixel size differences:** Ultrawide sensors sometimes have larger pixels, improving light collection per pixel and reducing photon shot noise.

3. **Exposure compensation:** The camera may have automatically increased exposure time for the ultrawide shot, allowing more light accumulation and improving the signal level.

4. **Sensor technology:** Newer/better sensor technology in the ultrawide module may have lower read noise or dark current, improving overall SNR.

The higher mean signal (32070 vs 23256) suggests the ultrawide collected significantly more light, which could be due to a larger aperture, longer exposure time, or both. Despite having 15% more absolute noise, the proportionally larger signal results in a better SNR.

### Quantization Noise

From the repository assignment, we also learned about quantization noise from analog-to-digital conversion:

```
σ_quantization = Δ / √12
```

where Δ is the quantization step size. For a 12-bit ADC (common in phone cameras):

- 2^12 = 4096 levels
- Quantization adds a small, uniform noise floor

This affects both cameras similarly, so it doesn't explain differences between them.

---

## 3. Practical Applications: Advantages and Disadvantages

### Main Camera

**Advantages:**

1. **Moderate FOV (68.92°)**

   - Balanced view for general photography
   - Natural perspective close to human vision
   - Versatile for most photography needs

2. **Natural perspective**

   - Minimal geometric distortion
   - Subject proportions appear natural
   - Suitable for portraits and everyday shots

3. **Better optical quality** (typically)
   - Usually the highest-quality lens in the phone
   - Sharper across the frame
   - Better color accuracy

**Disadvantages:**

1. **Limited FOV compared to ultrawide**

   - Cannot capture very wide scenes
   - Requires backing up in tight spaces
   - Less dramatic perspective effects

2. **Lower SNR in this comparison**
   - SNR of 5.79 vs ultrawide's 6.97
   - More visible noise in challenging conditions
   - May struggle more in low light

**Best for:**

- **Portrait photography:** Natural facial proportions, good subject isolation
- **General photography:** Everyday snapshots, social media, balanced compositions
- **When you need moderate "reach":** Capturing subjects at medium distances

---

### Ultrawide Camera

**Advantages:**

1. **Expansive FOV (104.81°)**

   - Captures **1.52× more of the scene** than main camera
   - Essential for cramped spaces (interiors, tight urban areas)
   - Creates sense of scale and space

2. **Better SNR in our test (6.97 vs 5.79)**

   - Higher signal levels despite wider FOV
   - Better noise performance than expected
   - Cleaner images in the tested conditions

3. **Dramatic perspective**

   - Exaggerated depth and distance
   - Creative "stretched" look
   - Emphasizes foreground elements

4. **Architectural/landscape photography**
   - Captures entire buildings without backing up
   - Shows sweeping vistas
   - Includes context and environment

**Disadvantages:**

1. **Geometric distortion**

   - Barrel distortion (straight lines curve)
   - Objects at edges appear stretched
   - Faces look distorted if too close

2. **Unnatural perspective**

   - Everything appears farther away
   - Can make subjects look small
   - Requires careful composition

3. **Edge quality degradation**
   - Image quality typically worse at frame edges
   - Sharpness and detail loss in corners
   - Vignetting possible

**Best for:**

- **Landscape photography:** Captures expansive scenes, dramatic skies, wide vistas
- **Architecture photography:** Entire buildings in frame, interior room shots
- **Group photos:** Fit more people in frame
- **Creative perspective:** Artistic shots with exaggerated depth

---

### Comparison with Telephoto Lenses

For reference, a **telephoto camera** (which you don't have but the assignment mentions) would show different characteristics:

**Telephoto lens (e.g., 77mm, f/2.8):**

- **Very narrow FOV** (~15°): Strong "zoom" effect, isolates distant subjects
- **Compressed perspective**: Makes foreground and background appear closer together
- **Typical smaller aperture**: Often f/2.8 or higher, leading to worse low-light performance
- **Best for**: Portrait photography (flattering compression), wildlife, sports, distant subjects
- **Disadvantages**: Very limited in tight spaces, requires more distance from subject

**The trade-off spectrum:**

```
Ultrawide (1.54mm) ← → Main (5.1mm) ← → Telephoto (77mm)
Wide FOV (105°)         Moderate (69°)      Narrow FOV (15°)
Dramatic depth          Natural             Compressed depth
Good in tight spaces    Versatile           Needs distance
Architectural/landscape General use         Portrait/wildlife
```

### Summary Table

| Feature        | Main Camera        | Ultrawide Camera         | (Telephoto for reference)   |
| -------------- | ------------------ | ------------------------ | --------------------------- |
| FOV            | Moderate (68.92°)  | Wide (104.81°)           | Narrow (~15°)               |
| Perspective    | Natural            | Dramatic                 | Compressed                  |
| Distortion     | Minimal            | Significant (barrel)     | Minimal (pincushion)        |
| SNR (measured) | 5.79               | 6.97 (better)            | Varies (often worse)        |
| Best use       | General, portraits | Landscapes, architecture | Distant subjects, portraits |
| Versatility    | High               | Specialized              | Specialized                 |

---

## Conclusion

This analysis reveals fundamental trade-offs in multi-camera smartphone systems:

### 1. Optical Properties

The inverse relationship between focal length and FOV creates distinct use cases. The main camera's moderate focal length (5.1mm) provides a natural perspective suitable for general photography, while the ultrawide's short focal length (1.54mm) captures expansive scenes with 1.52× wider coverage. (A telephoto lens, by contrast, would use a long focal length for a narrow, "zoomed-in" view.)

### 2. Noise Performance

Aperture size critically affects light collection and SNR. **Based on our data:** Contrary to typical expectations, the ultrawide camera demonstrated superior noise performance with 1.20× better SNR than the main camera. This suggests the ultrawide has compensatory features such as:

- A larger relative aperture (smaller f-number)
- Larger pixel sizes for better light collection
- Longer exposure compensation
- Advanced sensor technology

Despite having 15% more absolute noise (σ = 4600 vs 4013), the ultrawide's 38% higher signal level (μ = 32070 vs 23256) resulted in better overall SNR.

### 3. Application-Specific Design

Neither camera is universally "better" - each excels in specific scenarios:

- **Main camera:** Versatile workhorse for everyday photography, portraits, and natural perspective
- **Ultrawide camera:** Specialized tool for landscapes, architecture, and creative perspective with surprisingly good noise performance
- **(Telephoto camera - not tested):** Would excel at isolating distant subjects and portrait compression

### Key Insight

Modern smartphones leverage computational photography to mitigate each lens's weaknesses (correcting distortion, reducing noise through multi-frame averaging), but the fundamental physics of focal length, aperture, and sensor size continue to define each camera's strengths and limitations.

**Camera system design is about deliberate trade-offs** - optimizing different lenses for different tasks rather than attempting to create one "perfect" lens. The main vs ultrawide comparison demonstrates this principle: wide FOV and dramatic perspective vs. natural view, with the ultrawide surprisingly showing better SNR in our controlled test conditions.


# Part 3:

# Report on Convolution Filtering of Live Video Stream

## Experiment Setup
I captured live video from my laptop’s webcam, converted each frame to grayscale, and applied several different convolution filters implemented manually (without OpenCV built-in functions). I tested five filters: Box (average blur), Gaussian, Sobel (horizontal and vertical), Sharpening, and Emboss. The processed video stream was displayed side-by-side with the original webcam feed.

## Analysis and Observations

To understand the filters, I experimented with different real-life scenes:  
- **Hand movement in front of the camera** (dynamic object with edges).  
- **Well-lit object** (strong lighting contrast).  
- **Textured surface** (e.g., fabric or wall patterns).  

Below is the effect of each kernel.

---

### 1. Box Filter (3×3 Average)
- **Effect on video stream:** Produced a strong blur across the entire frame. When I waved my hand, the outline smeared, and small skin details disappeared. Text on a paper became harder to read.  
- **Kernel explanation:** Each element of the kernel is `1/9`. The output pixel becomes the average of its neighbors, reducing local intensity differences. This smooths noise but sacrifices detail.

---

### 2. Gaussian Filter (3×3)
- **Effect on video stream:** Similar to the box filter but smoother. My hand edges blurred less harshly, and noise in darker areas was reduced more naturally. Textured surfaces still retained their general shape.  
- **Kernel explanation:** The center value is largest, and surrounding values decrease with distance. This weighted average reduces random pixel noise while keeping more of the structural detail intact.

---

### 3. Sobel Filter (Horizontal & Vertical)
- **Effect on video stream:**  
  - Horizontal Sobel highlighted horizontal edges like the stripes on a book spine or lines on paper.  
  - Vertical Sobel highlighted vertical edges such as the sides of a door or the edges of my hand.  
  When I moved my hand quickly, the edges “glowed” strongly in one direction depending on the filter.  
- **Kernel explanation:** Sobel filters use positive values on one side and negative on the other (e.g., `[-1,0,1]`). This detects intensity changes in one direction, making edges stand out.

---

### 4. Sharpening Filter
- **Effect on video stream:** The image became more crisp and edgy. The outlines of my hand, face, and objects in the background were more pronounced. At the same time, noise also became sharper.  
- **Kernel explanation:** The kernel has a large positive center (e.g., `5`) and negative neighbors (`-1`). This amplifies the center pixel while subtracting the average of its surroundings. Positive center + negative neighbors → stronger edges and details.

---

### 5. Embossing Filter
- **Effect on video stream:** The video looked like a relief sculpture. My hand looked “raised” on one side with artificial shadows, and textured surfaces like a wall appeared as carved patterns.  
- **Kernel explanation:** Emboss kernels assign negative weights on one side and positive on the other (e.g., `[-2, -1, 0; -1, 1, 1; 0, 1, 2]`). This emphasizes diagonal intensity differences, creating a shading effect that mimics depth.

---

## Summary
- **Blurring kernels (Box, Gaussian):** Smooth out details and reduce noise by averaging neighboring pixels.  
- **Edge detection kernels (Sobel):** Highlight changes in intensity along specific directions.  
- **Sharpening kernels:** Emphasize edges by boosting the center pixel and suppressing neighbors.  
- **Embossing kernels:** Create an artificial 3D-like shading by offsetting pixel differences.  

These observations confirm how kernel structure (distribution of positive/negative weights and their magnitude) directly influences the visual effect in live video.
