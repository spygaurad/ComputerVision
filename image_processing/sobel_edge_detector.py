import numpy as np
from calculate_gradient import calculate_gradient

def sobel_edge_detector(img, threshold):
    """
    Apply Sobel edge detection with thresholding.
    
    Args:
        img: 2D numpy array (grayscale image)
        threshold: threshold value for binary edge map
    
    Returns:
        edge_map: binary edge map (255 for edges, 0 for non-edges)
    """
    # Calculate gradient magnitude
    grad_magnitude, _ = calculate_gradient(img)
    print(f"Gradient magnitude range: [{np.min(grad_magnitude):.2f}, {np.max(grad_magnitude):.2f}]")
    # Meaning of gradient magnitude values:
    # Low values (close to 0) indicate little change in intensity (flat regions)
    # High values indicate significant change in intensity (edges)
    
    # Apply binary threshold
    edge_map = np.zeros_like(grad_magnitude, dtype=np.uint8)
    edge_map[grad_magnitude > threshold] = 255
    
    return edge_map


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    # Load image
    image_path = 'fruits.png'
    # image_path = 'HappyFish.jpg'

    img = cv2.imread('images/'+image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image")
    else:
        # Calculate gradient magnitude for threshold selection
        grad_magnitude, _ = calculate_gradient(img)
        
        # The gradient magnitude values depend on the image content and the Sobel operator's
        # scaling. Values in the range of 0 to around 15.81 can be expected for typical images.
        # The maximum possible gradient magnitude for an 8-bit image using Sobel filters is around 1140,
        # but actual values are often much lower due to the nature of real-world images.

        thresholds = [6, 8,10]
        # thresholds = [10, 12,14] # for HappyFish.jpg

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Gradient magnitude
        axes[0, 1].imshow(grad_magnitude, cmap='gray')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].axis('off')
        
        # Histogram of gradient magnitudes
        axes[0, 2].hist(grad_magnitude.flatten(), bins=100, color='blue', alpha=0.7)
        axes[0, 2].set_title('Gradient Magnitude Histogram')
        axes[0, 2].set_xlabel('Magnitude')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Apply edge detection with different thresholds
        for idx, thresh in enumerate(thresholds):
            edge_map = sobel_edge_detector(img, thresh)
            
            row = 1
            col = idx if idx < 3 else idx - 3
            
            if idx < 3:
                axes[row, col].imshow(edge_map, cmap='gray')
                axes[row, col].set_title(f'Threshold = {thresh}')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/sobel_edge_detection_'+image_path.split('.')[0]+'.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Select optimal threshold (e.g., 50)
        optimal_threshold = 8  # 12 for HappyFish.jpg, 8 for fruits.png
        edge_map_final = sobel_edge_detector(img, optimal_threshold)
        
        # Display final result
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(edge_map_final, cmap='gray')
        axes[1].set_title(f'Sobel Edge Map (threshold={optimal_threshold})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/sobel_edge_final_'+image_path.split('.')[0]+'.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Gradient magnitude range: [{np.min(grad_magnitude):.2f}, {np.max(grad_magnitude):.2f}]")
        print(f"Mean gradient magnitude: {np.mean(grad_magnitude):.2f}")
        print(f"Selected threshold: {optimal_threshold}")
        
        edge_pixel_count = np.sum(edge_map_final == 255)
        total_pixels = edge_map_final.size
        edge_percentage = (edge_pixel_count / total_pixels) * 100
        print(f"Edge pixels: {edge_pixel_count} ({edge_percentage:.2f}%)")