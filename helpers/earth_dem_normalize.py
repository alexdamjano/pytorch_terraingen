from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 933120000

# Load the DEM (PNG or JPG image file)
def load_dem(dem_path):
    # Open the image using Pillow and convert to grayscale
    dem_image = Image.open(dem_path).convert('L')  # 'L' mode is for grayscale
    dem_array = np.array(dem_image)
    return dem_array

# Normalize the DEM and mask out values below sea level
def normalize_dem(dem_array, sea_level_threshold=0, sensitivity=1.0):
    # Mask out values below sea level (set them to black)
    dem_array[dem_array < sea_level_threshold] = 0
    
    # Find the minimum and maximum values of the DEM above sea level
    min_dem = np.min(dem_array[dem_array > sea_level_threshold])  # Ignore sea level
    max_dem = np.max(dem_array[dem_array > sea_level_threshold])  # Ignore sea level

    # Apply sensitivity adjustment (how much of the range you want to use for re-coloring)
    sensitivity_factor = sensitivity * (max_dem - min_dem)

    # Normalize the DEM to range [0, 255] for values above sea level
    normalized_dem = (dem_array - min_dem) / sensitivity_factor  # Normalize to [0,1]
    normalized_dem = np.clip(normalized_dem * 255, 0, 255)  # Scale to [0, 255] and clip out-of-bound values

    return np.uint8(normalized_dem)

# Save the result as PNG
def save_normalized_dem(dem_array, output_path):
    normalized_image = Image.fromarray(dem_array)
    normalized_image.save(output_path)
    print(f"Normalized DEM saved as {output_path}")

# Example usage
input_dem_path = "big_pics/earth_DEM2.png"  # Path to your large DEM image (PNG)
output_png_path = "big_pics/normalized_dem.png"  # Output PNG file

# Load the DEM
dem_array = load_dem(input_dem_path)

# Normalize the DEM to match your map's color scheme (adjust sensitivity and sea level threshold if needed)
normalized_dem = normalize_dem(dem_array, sea_level_threshold=127, sensitivity=0.65)  # Adjust sensitivity

# Save the normalized DEM as a PNG
save_normalized_dem(normalized_dem, output_png_path)


