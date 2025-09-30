import cv2
import numpy as np
import os

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


def main():
    input_path = "./images/bright_blobs.png"
    # input_path = "./images/room.png"

    filter_name = "gaussian"  # choose from get_filters()
    output_folder = "./outputs/out_conv"
    os.makedirs(output_folder, exist_ok=True)

    # Read image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read {input_path}")
        return
    
    # Get kernel
    filters = get_filters()
    if filter_name not in filters:
        print(f"Error: Filter {filter_name} not found")
        return
    kernel = filters[filter_name]
    
    # Apply convolution
    print("Applying convolution...")
    result = apply_convolution(img, kernel)
    print("Convolution done.")

    # Construct output filename
    base, ext = os.path.splitext(os.path.basename(input_path))
    output_path = f"{output_folder}/{base}_{filter_name}.png"

    # Concatenate original and result side by side
    combined = np.hstack((img, result))

    # Show images
    cv2.imshow("Original (left) vs Filtered (right)", combined)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    # Save combined result
    cv2.imwrite(output_path, combined)
    print(f"Saved side-by-side result to {output_path}")

if __name__ == "__main__":
    main()