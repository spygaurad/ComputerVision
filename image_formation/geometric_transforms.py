import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def read_image(image_path):
    """Reads an image from a file path."""
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return image

# Opencv cuts images during transformations, so we pad them first and unpad after transformation
def pad_image(image, pad_y, pad_x, border_value=(0,0,0)):
    h, w = image.shape[:2]
    padded = cv.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x,
        cv.BORDER_CONSTANT, value=border_value
    )
    return padded, (h, w, pad_y, pad_x)

#Unpad to original size
def unpad_image(image, original_info):
    h, w, pad_y, pad_x = original_info
    return image[pad_y:pad_y+h, pad_x:pad_x+w]


def perform_translation(image, tx, ty):
    """Translates the image by (tx, ty)."""
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def perform_rotation(image, angle, center=None, scale=1.0):
    """Rotates the image by a given angle around a center point."""
    rows, cols = image.shape[:2]
    if center is None:
        center = (cols // 2, rows // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def perform_shearing(image, shear_x=0.0, shear_y=0.0):
    """Shears the image by given shear factors along x and y axes."""
    rows, cols = image.shape[:2]
    shear_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    sheared_image = cv.warpAffine(image, shear_matrix, (cols, rows))
    return sheared_image

def perform_affine_transformation(image, src_points, dst_points):
    """Applies an affine transformation defined by source and destination points."""
    rows, cols = image.shape[:2]
    affine_matrix = cv.getAffineTransform(np.float32(src_points), np.float32(dst_points))
    affine_transformed_image = cv.warpAffine(image, affine_matrix, (cols, rows))
    return affine_transformed_image

def perform_scaling(image, scale_x, scale_y):
    """Scales the image by (scale_x, scale_y)."""
    rows, cols = image.shape[:2]
    scaled_image = cv.resize(image, (int(cols * scale_x), int(rows * scale_y)))
    return scaled_image

def perform_scaling_with_padding(image, scale_x, scale_y):
    """Scales the image and pastes it onto a black canvas of the original size."""
    rows, cols = image.shape[:2]
    new_cols, new_rows = int(cols * scale_x), int(rows * scale_y)
    scaled_image = cv.resize(image, (new_cols, new_rows))
    # Create a black canvas
    canvas = np.zeros_like(image)
    # Compute top-left corner for centering
    x_offset = (cols - new_cols) // 2
    y_offset = (rows - new_rows) // 2
    # Paste scaled image onto the canvas
    canvas[y_offset:y_offset+new_rows, x_offset:x_offset+new_cols] = scaled_image
    return canvas

def perform_perspective_transformation(image, src_points, dst_points):
    """Applies a perspective transformation defined by source and destination points."""
    rows, cols = image.shape[:2]
    perspective_matrix = cv.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    perspective_transformed_image = cv.warpPerspective(image, perspective_matrix, (cols, rows))
    return perspective_transformed_image

def chain_transformations(image, transformations):
    """Chains multiple transformations on the image."""
    transformed_image = image.copy()
    for transform in transformations:
        transformed_image = transform(transformed_image)
    return transformed_image

# Matplotlib Visualize original, provided transformed and my transformed
def visualize_transformations(original, provided_transformed, my_transformed, save_path=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Provided Transformed Image')
    plt.imshow(cv.cvtColor(provided_transformed, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('My Transformed Image')
    plt.imshow(cv.cvtColor(my_transformed, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    image_path = 'images/original_image.jpg'  # input image path
    transformed_image_path = 'images/transformed_image.jpg' # provided transformed image path

    original_image = read_image(image_path) # Read the original image
    original_transformed_image = read_image(transformed_image_path)  # Replace with your provided transformed image path

    tx1, ty1 = -200, -300  # Translation values
    tx2 ,ty2 = 120, 120
    angle = -45  # Rotation angle
    shear_x = 0.4  # Shear factor along x-axis
    shear_y = 0.6   # Shear factor along y-axis

    pad_x = 300
    pad_y = 300

    padded_image, original_info = pad_image(original_image, pad_y, pad_x) # Pad the image to avoid cropping during transformations
    transformations = [
            lambda img: perform_scaling_with_padding(img, scale_x=0.5, scale_y=0.9), #scale down the image and pad to original size
            lambda img: perform_translation(img, tx=tx1, ty=ty1), #translate
            lambda img: perform_shearing(img, shear_x=shear_x, shear_y=shear_y), #shear along both axes
            lambda img: unpad_image(img, original_info), # Unpad to original size
            lambda img: perform_translation(img, tx=tx2, ty=ty2), #translate


    ]
    transformed_image = chain_transformations(padded_image, transformations) # Apply chained transformations defined above
    #To visualize and save, use:
    # visualize_transformations(original_image, original_transformed_image, transformed_image, save_path='outputs/geometric_transforms.png')    
    visualize_transformations(original_image, original_transformed_image, transformed_image)    
    
    print("Applied Transformations: Padding, Scaling, Translation, Shearing, UnPadding and Translation")