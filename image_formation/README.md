#Exercise 1
- OpenCV cuts the original image if one of the transformation shifts the image outside boundary.
- I padded the image to a larger size and applied transformations.
- Transformations applied : Scaling with padding (Scaling reduces image size, so padding to input size), Translation, Shearing
- I unpadded the image and applied translation to move it to desired position.
- These transformers are chained in python using lambda function.


#Exercise 2
##Thin Lens Law and F Number Plots

###Task 1:

In the figure:
- Dotted straight lines mark the focal length positions for each lens
- Solid curved lines show how the image distance zi changes as the object distance zo changes, for different focal lengths.

####Key Observations:
- Lenses with longer focal lengths (e.g. 200 mm) require the image sensor to be placed much farther away from the lens to focus on nearby objects than lenses with short focal lengths (e.g. 3 mm).
- As the object distance 𝑧o approaches the focal length f from right, the required image distance zi increases rapidly and tends toward infinity — the curve becomes vertical at the dotted line
- When the object is very far away (large 𝑧0), all curves flatten out near zi​≈f, meaning distant objects focus at approximately the focal length.


###Task 2:

a)

####Key Takeways:
- This plot shows how the aperture diameter D increases linearly as the focal length 
f increases for different f-numbers (N).
- For any given f-number, D=f/N — so longer lenses require physically wider apertures to maintain the same f-number.
- Lower f-numbers (like f/1.4) have steeper slopes, meaning they need much larger diameters than higher f-numbers (like f/4) at the same focal length.


b)
Maximum aperture diameters (D = f / N)
  24mm f/1.4: D ≈ 17.14 mm
  50mm f/1.8: D ≈ 27.78 mm
  70–200mm f/2.8: 70mm → 25.00 mm; 200mm → 71.43 mm
  400mm f/2.8: D ≈ 142.86 mm
  600mm f/4.0: D ≈ 150.00 mm

#Exercise 3
- Issue in question, the n in quantize section should be n=2^num_bits. We have 8 levels, not just 3.


What do you think a reasonable sampling frequency should be to capture the true shape of the
signal? What should be done to minimize error?

- By shifting the sampling start point. (When sampling the original signal, if initiated from 0, it often sampled at 0. This is called sampling phase alignment. So i slightly shifted the sampling start point from 0 to 0.08, which gave non 0 smaples)
t_sampled = np.linspace(0, duration, n_samples, endpoint=False)
- By increasing the number of quantization bits as shown in above figure.
- By increasing sampling frequency. According to Nyquist, the minimum sampling frequency should be greater than 2 times f. For this exercise, to see the phases clearly, i found 30 sampling frequency to be reasonable.



#Exercise 4

The orange dots are the sampled values after noise is added — they scatter above or below where they should be, showing random deviations from the true signal.

MSE – average squared difference between original and distorted samples → shows overall error power.
RMSE – square root of MSE → shows average error size (in signal units).
PSNR – ratio of signal peak to error power (in dB) → shows signal quality (higher = cleaner).

Sampling: fs=8 Hz, duration=2.0s, samples=16
Quantization: bits=3, levels=8, range=[-1.0, 1.0]
Noise: mean=0.0, std_dev=0.1 (scaled by signal magnitude)

--- Error vs CLEAN SAMPLES ---
Noisy samples:    MSE=0.044593  RMSE=0.211171  PSNR=13.51 dB
Quantized (noisy): MSE=0.060225  RMSE=0.245407  PSNR=12.20 dB


