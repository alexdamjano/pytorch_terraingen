import cv2
import os

# Define input and output directories
high_res_dir = "data/high_test"   # Folder containing high-res tiles
output_dir = "data/high_test"  # Folder for resized high-res tiles

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Target size for high-res tiles
target_size = (600, 600)  # Change this if needed

# Process each high-res tile
for filename in os.listdir(high_res_dir):
    if filename.endswith(".png"):
        filepath = os.path.join(high_res_dir, filename)
        
        # Load the high-res image
        high_res_tile = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # Resize it to the target size
        resized_tile = cv2.resize(high_res_tile, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Save the resized tile
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, resized_tile)
        
        print(f"Resized {filename} to {target_size}")

print("\n✅ Resizing complete! Check 'data/high_res_resized' for results.")
