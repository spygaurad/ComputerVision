import numpy as np
from utils import clip_image
import cv2
import matplotlib.pyplot as plt

def contrast_stretch(img, r_min, r_max):
    """
    Maps the intensity range [r_min, r_max] of an image to the full output range [0, 255] linearly.
    
    Args:
        img: 2D numpy array (grayscale image)
        r_min: minimum intensity value in img
        r_max: maximum intensity value in img
    
    Returns:
        new_img: contrast stretched image
    """
    # r maax must be greater than r_min
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min")
    
    # initiate output image
    new_img = np.zeros_like(img, dtype=np.float64)
    
    # Apply contrast stretch
    new_img = ((img - r_min) / (r_max - r_min)) * 255.0

    new_img = clip_image(new_img, 0, 255) # clip values to range [0, 255]
    
    # convert to uint8 because images are usually in this format, if not converted, matplotlib may not display correctly
    return new_img.astype(np.uint8) 

if __name__ == "__main__":
    
    # Load a low-contrast image
    img = cv2.imread('images/low_contrast.png', cv2.IMREAD_GRAYSCALE)
    
    actual_min = np.min(img)
    actual_max = np.max(img)
    
    print(f"Image intensity range: [{actual_min}, {actual_max}]")
    
    # Apply contrast stretch
    stretched = contrast_stretch(img, actual_min, actual_max)
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255) # if no vmin, vmax, auto scales to [actual_min, actual_max] which hides contrast issue
    plt.title(f'Original (range: [{actual_min}, {actual_max}])')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(stretched, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Contrast Stretched (range: [0, 255])')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/contrast_stretch_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Stretched image range: [{np.min(stretched)}, {np.max(stretched)}]")