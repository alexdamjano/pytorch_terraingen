import cv2
import os
import numpy as np

# Load the world map
world_map = cv2.imread("input_map.png", cv2.IMREAD_GRAYSCALE)

# Define tile size
tile_size = 200  # Choose 128, 256, etc.

# Create output directory
os.makedirs("data/low_res", exist_ok=True)

# Cut the world map into tiles
h, w = world_map.shape
tile_id = 0
for y in range(0, h, tile_size):
    for x in range(0, w, tile_size):
        tile = world_map[y:y+tile_size, x:x+tile_size]
        cv2.imwrite(f"data/low_res/tile_{tile_id}.png", tile)
        tile_id += 1
