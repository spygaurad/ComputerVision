import numpy as np

def median_filter(img, size=3):
    """
    Apply a median filter to remove noise from an image.
    Steps:
    1. For each pixel, get neighboring pixels in a window of given size
    2. Sort all values in the window
    3. Replace center pixel with the median value
    
    Args:
        img: 2D numpy array (grayscale image)
        size: size of the filter window
    
    Returns:
        new_img: filtered image
    """
    if size % 2 == 0:
        raise ValueError("Filter size must be odd")
    
    img_h, img_w = img.shape
    pad_size = size // 2
    
    # Pad the image to handle borders, edge padding adds border pixels into the padding
    padded = np.pad(img, pad_size, mode='edge')
    
    # Initialize output image
    new_img = np.zeros_like(img)
    
    # Apply median filter
    for i in range(img_h):
        for j in range(img_w):
            # Extract window
            window = padded[i:i+size, j:j+size] # extract window of size x size, taking from padded image, no index out of bounds

            # Flatten window, sort, and get median
            window_flat = window.flatten()
            window_sorted = np.sort(window_flat)
            median_val = window_sorted[len(window_sorted) // 2]
            
            new_img[i, j] = median_val
    
    return new_img


def add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    """
    Add salt and pepper noise to an image.
    
    Args:
        img: input image
        salt_prob: probability of salt noise (white pixels)
        pepper_prob: probability of pepper noise (black pixels)
    
    Returns:
        noisy: noisy image
    """
    noisy = img.copy()
    
    # Generate random noise, and convert it later into salt and pepper noise
    rand = np.random.random(img.shape)
    
    # Add salt (white) noise, if random value is less than salt_prob, set pixel to 255
    noisy[rand < salt_prob] = 255

    # add pepper noise, Could do rand < pepper_prob but then salt and pepper could overlap, so do rand > (1 - pepper_prob)
    # ignores values greater than salt_prob but less than (1 - pepper_prob)
    noisy[rand > (1 - pepper_prob)] = 0
    
    return noisy


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    # Load image
    image_name = 'tulips.png'
    # PSNR (Noisy): 14.98 dB
    # PSNR (3x3 Filter): 33.46 dB
    # PSNR (5x5 Filter): 30.34 dB

    # Load a low-contrast image
    img = cv2.imread('images/'+image_name, cv2.IMREAD_GRAYSCALE)

    # Add salt and pepper noise
    noisy = add_salt_pepper_noise(img, salt_prob=0.05, pepper_prob=0.05)
    
    # Apply median filter with different sizes
    filtered_3x3 = median_filter(noisy, size=3)
    filtered_5x5 = median_filter(noisy, size=5)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Image (Salt & Pepper)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(filtered_3x3, cmap='gray')
    axes[1, 0].set_title('Median Filter (3x3)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(filtered_5x5, cmap='gray')
    axes[1, 1].set_title('Median Filter (5x5)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/median_filter_result_'+image_name.split('.')[0]+'.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate PSNR for comparison
    def calculate_psnr(original, processed):
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    print(f"PSNR (Noisy): {calculate_psnr(img, noisy):.2f} dB")
    print(f"PSNR (3x3 Filter): {calculate_psnr(img, filtered_3x3):.2f} dB")
    print(f"PSNR (5x5 Filter): {calculate_psnr(img, filtered_5x5):.2f} dB")