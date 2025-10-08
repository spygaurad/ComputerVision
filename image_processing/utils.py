import numpy as np


def apply_convolution(image, filter):
    """
    Apply convolution to a single channel image using a square filter.
    Using zero-padding as boundary condition.
    Args:
        image: 2D numpy array (single channel)
        filter: 2D numpy array (square, odd-sized kernel)
    Returns:
        Filtered image as 2D numpy array
    """
    # Get dimensions
    img_height, img_width = image.shape
    filter_size = filter.shape[0]
    pad_size = filter_size // 2
    
    # Create padded image (zero padding)
    # If kernel size = 5, pad size = 2
    # np.pad pads 2 pixels on each side i.e. top, bottom, left, right
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    # Initialize output image
    output = np.zeros_like(image, dtype=np.float64)
    
    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            # Extract region of interest
            region = padded_image[i:i+filter_size, j:j+filter_size]
            # Apply filter (element-wise multiplication and sum)
            output[i, j] = np.sum(region * filter)
    
    # Clip values to valid range
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def normalize_image(image, min_val=0, max_val=255):
    """
    Normalize image values to a specified range.
    
    Args:
        image: 2D numpy array
        min_val: minimum output value
        max_val: maximum output value
    
    Returns:
        normalized: normalized image
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max - img_min == 0:
        return np.full_like(image, min_val)
    
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def clip_image(image, min_val=0, max_val=255):
    """
    Clip image values to a specified range.
    
    Args:
        image: numpy array
        min_val: minimum value
        max_val: maximum value
    
    Returns:
        clipped: clipped image
    """
    return np.clip(image, min_val, max_val)



def get_filters():
    return {
        'box': np.ones((3, 3)) / 9,
        'gaussian': np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]]) / 16,
        'sobel_horizontal': np.array([[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2,  1]]),
        'sobel_vertical': np.array([[-1,  0,  1],
                                    [-2,  0,  2],
                                    [-1,  0,  1]]),
        'sharpening': np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]]),
        'emboss': np.array([[-2, -1,  0],
                            [-1,  1,  1],
                            [ 0,  1,  2]])
    }

def get_sobel_kernels():
    # change dtype to np.float64
    sobel_horizontal = get_filters()['sobel_horizontal'].astype(np.float64)
    sobel_vertical = get_filters()['sobel_vertical'].astype(np.float64)
    return sobel_horizontal, sobel_vertical