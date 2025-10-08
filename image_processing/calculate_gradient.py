import numpy as np
from utils import apply_convolution, get_sobel_kernels

def calculate_gradient(img):
    """
    Calculate gradient magnitude and direction using Sobel operators.
    
    Args:
        img: 2D numpy array (grayscale image)
    
    Returns:
        grad_magnitude: gradient magnitude
        grad_angle: gradient direction in degrees [0, 360)
    """
    # Get Sobel kernels
    Sx, Sy = get_sobel_kernels()
    
    # Apply Sobel filters
    Gx = apply_convolution(img.astype(np.float64), Sx)
    Gy = apply_convolution(img.astype(np.float64), Sy)
    
    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Calculate gradient direction (in radians, then convert to degrees)
    grad_angle = np.arctan2(Gy, Gx)
    
    # Convert to degrees and ensure range [0, 360)
    grad_angle = np.degrees(grad_angle)
    grad_angle = (grad_angle + 360) % 360
    # grad_angle = grad_angle % 180  # Map to [0, 180) since direction is ambiguous
    
    return grad_magnitude, grad_angle


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from median_filter import add_salt_pepper_noise, median_filter
    
    # Load image
    image_name = 'tulips.png'

    # Load a low-contrast image
    img = cv2.imread('images/'+image_name, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image")
    else:
        # Clean image gradient
        grad_mag_clean, grad_angle_clean = calculate_gradient(img)
        
        # Add noise
        noisy = add_salt_pepper_noise(img, salt_prob=0.05, pepper_prob=0.05)
        
        # Gradient of noisy image
        grad_mag_noisy, grad_angle_noisy = calculate_gradient(noisy)
        
        # Apply median filter then calculate gradient
        filtered = median_filter(noisy, size=5)
        grad_mag_filtered, grad_angle_filtered = calculate_gradient(filtered)
        
        # Display results
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Row 1: Original images
        axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Noisy Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(filtered, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title('Filtered Image (5x5 Median)')
        axes[0, 2].axis('off')
        
        # Row 2: Gradient magnitudes
        axes[1, 0].imshow(grad_mag_clean, cmap='gray')
        axes[1, 0].set_title('Gradient Magnitude (Clean)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(grad_mag_noisy, cmap='gray')
        axes[1, 1].set_title('Gradient Magnitude (Noisy)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(grad_mag_filtered, cmap='gray')
        axes[1, 2].set_title('Gradient Magnitude (After Filtering)')
        axes[1, 2].axis('off')
        
        # Row 3: Gradient directions
        axes[2, 0].imshow(grad_angle_clean, cmap='hsv')
        axes[2, 0].set_title('Gradient Direction (Clean)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(grad_angle_noisy, cmap='hsv')
        axes[2, 1].set_title('Gradient Direction (Noisy)')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(grad_angle_filtered, cmap='hsv')
        axes[2, 2].set_title('Gradient Direction (After Filtering)')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/gradient_analysis_'+image_name.split('.')[0]+'.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nAnalysis of median filtering effect on gradient magnitude:")
        print(f"Clean gradient - Mean: {np.mean(grad_mag_clean):.2f}, Std: {np.std(grad_mag_clean):.2f}")
        print(f"Noisy gradient - Mean: {np.mean(grad_mag_noisy):.2f}, Std: {np.std(grad_mag_noisy):.2f}")
        print(f"Filtered gradient - Mean: {np.mean(grad_mag_filtered):.2f}, Std: {np.std(grad_mag_filtered):.2f}")
        #Median filtering reduces the standard deviation significantly, indicating noise reduction.
        # Why getting inf std?
        """
            arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
            Clean gradient - Mean: 6.22, Std: inf
            Noisy gradient - Mean: 6.02, Std: inf
            Filtered gradient - Mean: 5.45, Std: inf
            Filtered gradient magnitude is valid.
        """
