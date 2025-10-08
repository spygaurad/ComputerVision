import numpy as np

def calculate_histogram(img, bins):
    """
    Function to calculate histogram and normalized histogram 
    Args:
        img: 2D numpy array  of grayscale image
        bins: number of bins
    
    Returns:
        counts: histogram counts in each bin (1d array of size bins)
        dist: normalized histogram, counts/total_counts
    """
    # Initialize counts array
    counts = np.zeros(bins, dtype=np.int64)
    
    # Calculate bin width, 1 in this case
    bin_width = 256.0 / bins
    
    # Flatten image for 1D processing
    img_flat = img.flatten()
    
    # Count pixels in each bin
    for pixel_val in img_flat:
        # Determine which bin this pixel belongs to, 
        # if bin width was 10, pixel value 23 would go to bin index 2 (23/10 = 2.3 -> int() -> 2)
        bin_idx = int(pixel_val / bin_width)

        # clip values greater than max to last bin
        if bin_idx >= bins:
            bin_idx = bins - 1
        
        counts[bin_idx] += 1 # increment count for that bin
    
    # normalize
    total_pixels = img.size
    dist = counts.astype(np.float64) / total_pixels # convert to float for accurate division

    return counts, dist


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    # Load image
    img = cv2.imread('images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image")
    else:
        # Calculate histogram with 256 bins (one per intensity level)
        counts, dist = calculate_histogram(img, 256)
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Histogram (counts)
        axes[0, 1].bar(range(256), counts, width=1.0, color='black')
        axes[0, 1].set_title('Histogram (Counts)')
        axes[0, 1].set_xlabel('Intensity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xlim([0, 255])
        
        # Normalized histogram (distribution)
        axes[1, 0].bar(range(256), dist, width=1.0, color='blue')
        axes[1, 0].set_title('Normalized Histogram (Probability)')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].set_xlim([0, 255])
        
        # Cumulative distribution
        cumsum = np.cumsum(dist)
        axes[1, 1].plot(range(256), cumsum, color='red')
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_xlim([0, 255])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/histogram_analysis_cameraman.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Total pixel count: {np.sum(counts)}")
        print(f"Probability sum: {np.sum(dist):.6f}")