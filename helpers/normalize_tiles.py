import cv2
import os
import numpy as np

# Define input and output directories
input_dir = "data/low_res"      # Folder containing your DEM tiles
output_dir = "data/normalized"  # Folder to save normalized tiles

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all tiles in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # Ensure it's a PNG image
        filepath = os.path.join(input_dir, filename)

        # Load the tile in grayscale mode
        tile = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Normalize pixel values to range [0,1]
        normalized_tile = tile.astype(np.float32) / 255.0  

        # Save the normalized tile
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, (normalized_tile * 255).astype(np.uint8))

        # Print statistics for debugging
        print(f"{filename}: Min={normalized_tile.min()}, Max={normalized_tile.max()}")
