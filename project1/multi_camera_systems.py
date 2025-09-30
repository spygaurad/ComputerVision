import numpy as np
import rawpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_raw_as_2d(path):
    """
    Load DNG and return single-channel 2D raw sensor array (float32).
    This preserves the original Bayer pattern without demosaicing.
    
    Args:
        path: Path to the DNG file
        
    Returns:
        arr: 2D numpy array of raw sensor values (float32)
    """
    print(f"\nLoading raw image: {path}")
    with rawpy.imread(path) as raw:
        # raw.raw_image is the sensor data (2D). Cast to float for numeric ops.
        arr = raw.raw_image.astype(np.float32)
        
        """ 
        On digital image sensors (CMOS/CCD), the black level is the baseline 
        pixel value the camera electronics report when no light hits the pixel.
        Because the sensor + readout electronics add an offset, the raw pixel 
        values are not truly zero even in complete darkness.
        """
        try:
            # raw.black_level_per_channel is often an array like [bl, bl, bl, bl]
            bl = np.mean(raw.black_level_per_channel)
            print(f"  Black level: {bl:.2f}")
            arr = arr - bl
        except Exception as e:
            print(f"  Warning: Could not subtract black level: {e}")
            pass
        
        print(f"  Image shape: {arr.shape}")
        print(f"  Value range: [{arr.min():.2f}, {arr.max():.2f}]")
        
        return arr


def calculate_fov(focal_length, sensor_width):
    """
    Calculate the horizontal field of view in degrees.
    
    Formula: FOV = 2 * arctan(w / (2*f))
    where w = sensor width, f = focal length
    
    Args:
        focal_length: Focal length in mm
        sensor_width: Sensor width in mm
        
    Returns:
        fov: Field of view in degrees
    """
    fov_rad = 2 * np.arctan(sensor_width / (2 * focal_length))
    fov_deg = np.degrees(fov_rad)
    return fov_deg


def get_selected_region(coords):
    """
    Convert coordinate pairs to region dictionary.
    
    Args:
        coords: List of two (x, y) tuples
        
    Returns:
        Dictionary with x1, y1, x2, y2 keys
    """
    if len(coords) == 2:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        # Ensure x1 < x2 and y1 < y2
        return {
            'x1': min(x1, x2),
            'y1': min(y1, y2),
            'x2': max(x1, x2),
            'y2': max(y1, y2)
        }
    else:
        print("Region selection cancelled or incomplete.")
        return None


def calculate_noise_stats_2d(image, region_coords):
    """
    Calculate mean and standard deviation for a selected region.
    Works with 2D raw sensor data (single channel).
    
    Args:
        image: 2D numpy array of raw sensor values
        region_coords: Dictionary with 'x1', 'y1', 'x2', 'y2'
        
    Returns:
        stats: Dictionary containing mean and std
    """
    x1, y1 = region_coords['x1'], region_coords['y1']
    x2, y2 = region_coords['x2'], region_coords['y2']
    
    # Extract region (note: image indexing is [y, x])
    region = image[y1:y2, x1:x2]
    
    # Calculate statistics for the raw sensor data
    mean_val = np.mean(region)
    std_val = np.std(region)
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'snr': mean_val / std_val if std_val > 0 else float('inf'),
        'region_size': region.shape,
        'num_pixels': region.size
    }
    
    return stats


def print_results(camera_name, fov, stats):
    """Print analysis results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{camera_name} CAMERA RESULTS")
    print(f"{'='*60}")
    print(f"\nField of View: {fov:.2f}°")
    print(f"\nNoise Analysis (Raw Sensor Data)")
    print(f"Region size: {stats['region_size'][0]}×{stats['region_size'][1]} pixels ({stats['num_pixels']} total)")
    print(f"{'-'*60}")
    print(f"Mean (μ):           {stats['mean']:.2f}")
    print(f"Std Dev (σ):        {stats['std']:.4f}")
    print(f"Signal-to-Noise:    {stats['snr']:.2f}")
    print(f"{'='*60}")


def display_task():
    """Display task header."""
    print("="*60)
    print("MULTI-CAMERA SYSTEM ANALYSIS")
    print("Main Camera vs. Ultrawide Camera")
    print("="*60)


def display_fov_differences(main_focal, main_sensor, main_fov, 
                           ultra_focal, ultra_sensor, ultra_fov):
    """Display FOV comparison between cameras."""
    print(f"\nMain Camera:")
    print(f"  Focal Length: {main_focal} mm")
    print(f"  Sensor Width: {main_sensor} mm")
    print(f"  Horizontal FOV: {main_fov:.2f}°")
    
    print(f"\nUltrawide Camera:")
    print(f"  Focal Length: {ultra_focal} mm")
    print(f"  Sensor Width: {ultra_sensor} mm")
    print(f"  Horizontal FOV: {ultra_fov:.2f}°")
    
    print(f"\nFOV Difference: {abs(ultra_fov - main_fov):.2f}° " 
          f"(Ultrawide is {ultra_fov/main_fov:.2f}x wider)")


def visualize_raw_image(image, title="Raw Sensor Data"):
    """
    Display raw sensor data as grayscale image.
    
    Args:
        image: 2D numpy array
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray', vmin=0, vmax=np.percentile(image, 99))
    plt.title(title)
    plt.colorbar(label='Pixel Value')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_region_selection(image, region_coords, title="Selected Region"):
    """
    Display image with selected region highlighted.
    
    Args:
        image: 2D numpy array
        region_coords: Dictionary with region coordinates
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray', vmin=0, vmax=np.percentile(image, 99))
    
    # Draw rectangle around selected region
    x1, y1 = region_coords['x1'], region_coords['y1']
    x2, y2 = region_coords['x2'], region_coords['y2']
    width = x2 - x1
    height = y2 - y1
    
    rect = Rectangle((x1, y1), width, height, 
                     linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_title(f"{title}\nRegion: ({x1},{y1}) to ({x2},{y2})")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    # ========================================================================
    # CAMERA SPECIFICATIONS
    # ========================================================================
    # TODO: Update these values for your specific phone model
    MAIN_FOCAL_LENGTH = 5.1  # mm (example: iPhone 15 Pro main camera)
    MAIN_SENSOR_WIDTH = 7  # mm (example)

    ULTRAWIDE_FOCAL_LENGTH = 1.54  # mm (example: iPhone 15 Pro ultrawide)
    ULTRAWIDE_SENSOR_WIDTH = 4 # mm (example)

    # ========================================================================
    # IMAGE PATHS
    # ========================================================================
    MAIN_IMAGE_PATH = "./images/sheep_main_lens.dng"
    ULTRAWIDE_IMAGE_PATH = "./images/sheep_wide_lens.dng"

    # ========================================================================
    # UNIFORM REGION COORDINATES
    # ========================================================================
    # Format: [(x1, y1), (x2, y2)] - top-left and bottom-right corners
    main_image_uniform_patch_coord = [(1373, 1599), (1476, 1672)]
    ultrawide_image_uniform_patch_coord = [(1414, 1446), (1522, 1512)]

    # ========================================================================
    # STEP 1: FIELD OF VIEW CALCULATION
    # ========================================================================
    display_task()
    
    print("\n" + "="*60)
    print("STEP 1: FIELD OF VIEW CALCULATION")
    print("="*60)

    main_fov = calculate_fov(MAIN_FOCAL_LENGTH, MAIN_SENSOR_WIDTH)
    ultrawide_fov = calculate_fov(ULTRAWIDE_FOCAL_LENGTH, ULTRAWIDE_SENSOR_WIDTH)
    
    display_fov_differences(
        MAIN_FOCAL_LENGTH, MAIN_SENSOR_WIDTH, main_fov,
        ULTRAWIDE_FOCAL_LENGTH, ULTRAWIDE_SENSOR_WIDTH, ultrawide_fov
    )

    # ========================================================================
    # STEP 2: LOAD RAW IMAGES
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: LOADING RAW IMAGES (NO POST-PROCESSING)")
    print("="*60)
    
    main_image = load_raw_as_2d(MAIN_IMAGE_PATH)
    ultrawide_image = load_raw_as_2d(ULTRAWIDE_IMAGE_PATH)

    # Optional: Visualize the raw images
    # Uncomment the lines below to see the images
    # visualize_raw_image(main_image, "Main Camera - Raw Sensor Data")
    # visualize_raw_image(ultrawide_image, "Ultrawide Camera - Raw Sensor Data")

    # ========================================================================
    # STEP 3: NOISE ANALYSIS - MAIN CAMERA
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: NOISE ANALYSIS - MAIN CAMERA")
    print("="*60)
    
    main_region = get_selected_region(main_image_uniform_patch_coord)
    
    if main_region is None:
        print("Error: Invalid main camera region coordinates")
        return
    
    main_stats = calculate_noise_stats_2d(main_image, main_region)
    print_results("MAIN", main_fov, main_stats)
    
    # Optional: Visualize the selected region
    # visualize_region_selection(main_image, main_region, "Main Camera - Selected Region")

    # ========================================================================
    # STEP 4: NOISE ANALYSIS - ULTRAWIDE CAMERA
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: NOISE ANALYSIS - ULTRAWIDE CAMERA")
    print("="*60)
    
    ultrawide_region = get_selected_region(ultrawide_image_uniform_patch_coord)
    
    if ultrawide_region is None:
        print("Error: Invalid ultrawide camera region coordinates")
        return
    
    ultrawide_stats = calculate_noise_stats_2d(ultrawide_image, ultrawide_region)
    print_results("ULTRAWIDE", ultrawide_fov, ultrawide_stats)
    
    # Optional: Visualize the selected region
    # visualize_region_selection(ultrawide_image, ultrawide_region, 
    #                           "Ultrawide Camera - Selected Region")

    # ========================================================================
    # STEP 5: COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nOptical Properties:")
    print(f"  Main FOV: {main_fov:.2f}° | Ultrawide FOV: {ultrawide_fov:.2f}°")
    print(f"  Ultrawide captures {ultrawide_fov/main_fov:.2f}x more of the scene")
    
    print(f"\nNoise Comparison (Raw Sensor Data):")
    print(f"  Main Camera:")
    print(f"    Mean (μ):     {main_stats['mean']:.2f}")
    print(f"    Std Dev (σ):  {main_stats['std']:.4f}")
    print(f"    SNR (μ/σ):    {main_stats['snr']:.2f}")
    
    print(f"\n  Ultrawide Camera:")
    print(f"    Mean (μ):     {ultrawide_stats['mean']:.2f}")
    print(f"    Std Dev (σ):  {ultrawide_stats['std']:.4f}")
    print(f"    SNR (μ/σ):    {ultrawide_stats['snr']:.2f}")
    
    # Noise comparison
    noise_ratio = ultrawide_stats['std'] / main_stats['std']
    print(f"\n  Noise Ratio (σ_ultrawide / σ_main): {noise_ratio:.3f}")
    if noise_ratio > 1:
        print(f"  → Ultrawide has {noise_ratio:.2f}x more noise than Main")
    else:
        print(f"  → Main has {1/noise_ratio:.2f}x more noise than Ultrawide")
    
    # SNR comparison
    snr_ratio = ultrawide_stats['snr'] / main_stats['snr']
    print(f"\n  SNR Ratio (SNR_ultrawide / SNR_main): {snr_ratio:.3f}")
    if snr_ratio > 1:
        print(f"  → Ultrawide has {snr_ratio:.2f}x better SNR than Main")
    else:
        print(f"  → Main has {1/snr_ratio:.2f}x better SNR than Ultrawide")
    

if __name__ == "__main__":
    main()