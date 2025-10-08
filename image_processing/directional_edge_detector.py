import numpy as np
from calculate_gradient import calculate_gradient

def directional_edge_detector(img, direction_range):
    """
    Detect edges in a specific direction range.
    
    Args:
        img: 2D numpy array (grayscale image)
        direction_range: tuple (min_angle, max_angle) in degrees
                        e.g., (40, 50) for roughly 45-degree edges
    
    Returns:
        edge_directional_map: binary map showing edges in the specified direction
    """
    # Calculate gradient magnitude and direction
    grad_magnitude, grad_angle = calculate_gradient(img)
    
    min_angle, max_angle = direction_range
    
    # Initialize output map
    edge_directional_map = np.zeros_like(grad_magnitude, dtype=np.uint8)
    
    if min_angle <= max_angle:
        # for angles in range without wrapping, example works for 0-10, 40-50, 85-95, 130-140
        # Wont work for ranges 359-370 which is essentially 359-10. In that case, use the else part.
        mask = (grad_angle >= min_angle) & (grad_angle <= max_angle)
    else:
        # Wrapping case when ranges have to cross 0 because min_angle > max_angle, eg min angle=350, max_angle=10.
        # What this 
        mask = (grad_angle >= min_angle) | (grad_angle <= max_angle)
    

    # Set edges in the specified direction to 255
    edge_directional_map[mask] = 255
    
    return edge_directional_map


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from sobel_edge_detector import sobel_edge_detector
    
    # Load image
    image_name = "fruits.png"
    img = cv2.imread('images/' + image_name, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not load image")
    else:
        # Calculate gradient for visualization
        grad_magnitude, grad_angle = calculate_gradient(img)
        
        # Define directional ranges
        # 0° = horizontal right, 90° = vertical up, 180° = horizontal left, 270° = vertical down
        # 45° = diagonal from lower-left to upper-right
        directions = {
            'Horizontal (0°)': (0, 10),
            'Diagonal 45°': (40, 50),
            'Vertical (90°)': (85, 95),
            'Diagonal 135°': (130, 140)
        }
        
        # Apply Sobel edge detector (magnitude-based)
        sobel_edges = sobel_edge_detector(img, threshold=8)
        
        # Apply Canny edge detector (OpenCV implementation)
        canny_edges = cv2.Canny(img, threshold1=50, threshold2=150)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 12))
        
        # Original image
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Gradient magnitude
        ax2 = plt.subplot(3, 3, 2)
        ax2.imshow(grad_magnitude, cmap='gray')
        ax2.set_title('Gradient Magnitude')
        ax2.axis('off')
        
        # Gradient direction (color-coded)
        ax3 = plt.subplot(3, 3, 3)
        im = ax3.imshow(grad_angle, cmap='hsv', vmin=0, vmax=360)
        ax3.set_title('Gradient Direction')
        ax3.axis('off')
        # using colorbar to indicate angle values because interpreting hsv colors is not intuitive
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Degrees')
        
        # Directional edge detections
        plot_idx = 4
        for direction_name, direction_range in directions.items():
            directional_edges = directional_edge_detector(img, direction_range)
            
            ax = plt.subplot(3, 3, plot_idx)
            ax.imshow(directional_edges, cmap='gray')
            ax.set_title(f'{direction_name}\n[{direction_range[0]}°-{direction_range[1]}°]')
            ax.axis('off')
            
            plot_idx += 1
        
        # Sobel edge detector (magnitude-based)
        ax_sobel = plt.subplot(3, 3, 8)
        ax_sobel.imshow(sobel_edges, cmap='gray')
        ax_sobel.set_title('Sobel Edge Detector\n(Magnitude-based, threshold=50)')
        ax_sobel.axis('off')
        
        # Canny edge detector
        ax_canny = plt.subplot(3, 3, 9)
        ax_canny.imshow(canny_edges, cmap='gray')
        ax_canny.set_title('Canny Edge Detector\n(OpenCV, 50-150)')
        ax_canny.axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/directional_edge_detection_' + image_name.split('.')[0] + '.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Comparison figure for analysis
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sobel magnitude-based
        axes[0].imshow(sobel_edges, cmap='gray')
        axes[0].set_title('Sobel (Magnitude Only)')
        axes[0].axis('off')
        
        # Directional (45 degrees)
        directional_45 = directional_edge_detector(img, (40, 50))
        axes[1].imshow(directional_45, cmap='gray')
        axes[1].set_title('Directional (45° Edges)')
        axes[1].axis('off')
        
        # Canny
        axes[2].imshow(canny_edges, cmap='gray')
        axes[2].set_title('Canny Edge Detector')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/edge_detection_comparison_' + image_name.split('.')[0] + '.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        """
        === Analysis ===
        1. Sobel Edge Detector (Magnitude-based):
        - Detects edges based only on gradient magnitude
        - Captures all edges regardless of direction
        - Simple thresholding produces thick edges
        - No edge thinning or non-maximum suppression

        2. Directional Edge Detector:
        - Filters edges by gradient direction
        - Useful for detecting specific orientations (e.g., 45° diagonal edges)
        - Can isolate vertical, horizontal, or diagonal features
        - Produces directional-specific edge maps
        
        3. Canny Edge Detector:
        - Multi-stage algorithm with sophisticated processing
        - Includes Gaussian smoothing, non-maximum suppression, and hysteresis thresholding
        - Produces thin, well-connected edges
        - Better at suppressing noise and weak edges
        - Generally superior for practical edge detection tasks
        """
 
        # Calculate edge statistics
        sobel_count = np.sum(sobel_edges == 255)
        directional_count = np.sum(directional_45 == 255)
        canny_count = np.sum(canny_edges == 255)
        total_pixels = img.size
        
        print("\n=== Edge Pixel Statistics ===")
        print(f"Sobel: {sobel_count} pixels ({100*sobel_count/total_pixels:.2f}%)")
        print(f"Directional (45°): {directional_count} pixels ({100*directional_count/total_pixels:.2f}%)")
        print(f"Canny: {canny_count} pixels ({100*canny_count/total_pixels:.2f}%)")