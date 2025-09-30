import os
import argparse
from glob import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt

def load_raw_as_2d(path):
    """Load DNG and return single-channel 2D raw sensor array (float32)."""
    with rawpy.imread(path) as raw:
        # raw.raw_image is the sensor data (2D). Cast to float for numeric ops.
        arr = raw.raw_image.astype(np.float32)
        """ 
        On digital image sensors (CMOS/CCD), the black level is the baseline pixel value the camera electronics report when no light hits the pixel.
        Because the sensor + readout electronics add an offset, the raw pixel values are not truly zero even in complete darkness.
        """
        # try:
        #     # raw.black_level_per_channel is often an array like [bl, bl, bl, bl]
        #     bl = np.mean(raw.black_level_per_channel)
        #     arr = arr - bl
        # except Exception:
        #     pass
        return arr

def center_patch(arr, patch_size=100):
    """Return center patch (square) of patch_size x patch_size from 2D array."""
    h, w = arr.shape
    ps = int(patch_size)
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - ps // 2)
    x0 = max(0, cx - ps // 2)
    y1 = min(h, y0 + ps)
    x1 = min(w, x0 + ps)
    return arr[y0:y1, x0:x1]

def analyze_file(path, patch_size=100, bins=200, hist_range=None, outdir="./outputs/out_hist"):
    os.makedirs(outdir, exist_ok=True)
    arr = load_raw_as_2d(path)
    print(f"File: {os.path.basename(path)}")
    print(f"  dtype: {arr.dtype}, shape: {arr.shape}")
    print(f"  min: {float(arr.min()):.3f}, max: {float(arr.max()):.3f}")

    patch = center_patch(arr, patch_size=patch_size)
    mu = float(np.mean(patch))
    sigma = float(np.std(patch, ddof=1))
    print(f"  patch (size {patch.shape}) mean: {mu:.3f}, std: {sigma:.3f}")

    # Histogram of the patch (more detail where noise lives)
    plt.figure(figsize=(6,4))
    if hist_range is None:
        # choose tight range around patch mean ± 6 sigma to focus on noise distribution
        hist_min = mu - 6*sigma
        hist_max = mu + 6*sigma
    else:
        hist_min, hist_max = hist_range
    plt.hist(patch.flatten(), bins=bins, range=(hist_min, hist_max))
    plt.title(f"Histogram: {os.path.basename(path)}\npatch mean={mu:.2f}, std={sigma:.2f}")
    plt.xlabel("Raw sensor value (black subtracted if available)")
    plt.ylabel("Count")
    outpath = os.path.join(outdir, os.path.basename(path) + ".png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"saved histogram to {outpath}\n")
    return {
        "file": path,
        "shape": arr.shape,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "patch_shape": patch.shape,
        "patch_mean": mu,
        "patch_std": sigma,
        "hist_range": (hist_min, hist_max)
    }

def main():
    paths = [
        "./images/dark_1.dng",
        "./images/dark_2.dng",
        "./images/dark_3.dng",
    ]
    out_path = "./outputs/out_hist"
    no_of_bins = 200
    patch_size = 100

    results = []
    for p in paths:
        try:
            r = analyze_file(p, patch_size=patch_size, bins=no_of_bins, outdir=out_path)
            results.append(r)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    # Print a short summary
    print("Summary:")
    for r in results:
        print(f"- {os.path.basename(r['file'])}: patch mean={r['patch_mean']:.3f}, std={r['patch_std']:.3f}")

if __name__ == "__main__":
    main()