import numpy as np
from calculate_histogram import calculate_histogram

def equalize_histogram(img, nbins=256):
    """
    Histogram equalization.
    Steps:
    1. Calculate histogram and normalize it
    2. Calculate cdf
    3. scale original pixels with cdf and 255
    
    Args:
        img: 2D numpy array of grayscale image
    
    Returns:
        new_img: histogram equalized image
    """
    # Calculate histogram with 256 bins, 1 bin for 1 pixel value
    counts, dist = calculate_histogram(img, nbins)
    
    # Calculate cdf
    cdf = np.cumsum(dist) # shape is no. of bins

    new_img = np.zeros_like(img, dtype=np.uint8)
    bin_width = 256 / nbins # for 256 bins, each bin covers 1 pixel value, for nbin = 1

    # Map each pixel to its equalized value
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_value = img[i, j]
            
            # Determine which bin this pixel belongs to
            bin_idx = int(pixel_value / bin_width)
            
            # Handle edge case where pixel value is exactly 255
            if bin_idx >= nbins:
                bin_idx = nbins - 1
            
            # Map using CDF: scale to [0, 255]
            new_img[i, j] = int(cdf[bin_idx] * 255)
    
    return new_img
    


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from contrast_stretch import contrast_stretch
    
    # image_name = 'low_contrast.png'
    image_name = 'cameraman.bmp'

    # Load a low-contrast image
    img = cv2.imread('images/'+image_name, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image")
    else:
        # Apply contrast stretch
        actual_min = np.min(img)
        actual_max = np.max(img)
        stretched = contrast_stretch(img, actual_min, actual_max)
        
        # Apply histogram equalization
        equalized = equalize_histogram(img)
        
        # Calculate histograms for comparison
        counts_orig, dist_orig = calculate_histogram(img, 256)
        counts_stretch, dist_stretch = calculate_histogram(stretched, 256)
        counts_eq, dist_eq = calculate_histogram(equalized, 256)
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Original image and histogram
        axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].bar(range(256), counts_orig, width=1.0, color='black')
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xlim([0, 255])
        
        # Contrast stretched image and histogram
        axes[0, 1].imshow(stretched, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Contrast Stretched')
        axes[0, 1].axis('off')
        
        axes[1, 1].bar(range(256), counts_stretch, width=1.0, color='blue')
        axes[1, 1].set_title('Stretched Histogram')
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xlim([0, 255])
        
        # Equalized image and histogram
        axes[0, 2].imshow(equalized, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title('Histogram Equalized')
        axes[0, 2].axis('off')
        
        axes[1, 2].bar(range(256), counts_eq, width=1.0, color='green')
        axes[1, 2].set_title('Equalized Histogram')
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_xlim([0, 255])
        
        plt.tight_layout()
        plt.savefig('outputs/histogram_equalization_'+image_name.split('.')[0]+'.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Analysis:")
        print(f"Original - Range: [{np.min(img)}, {np.max(img)}], Std: {np.std(img):.2f}")
        print(f"Stretched - Range: [{np.min(stretched)}, {np.max(stretched)}], Std: {np.std(stretched):.2f}")
        print(f"Equalized - Range: [{np.min(equalized)}, {np.max(equalized)}], Std: {np.std(equalized):.2f}")