import os
import cv2
import numpy as np
import shutil  # Used to copy files

# Directories
low_res_dir = "data/normalized"  # Low-res tiles
high_res_dir = "data/high_test"  # High-res tiles (original)
output_dir = "data/high_res"     # New folder for renamed copies

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# ORB feature detector
orb = cv2.ORB_create()

# Function to compute ORB descriptors
def compute_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

# Function to compute histogram similarity (backup matcher)
def compute_histogram_similarity(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # Higher means better match

# Get sorted lists of files
low_res_files = sorted([f for f in os.listdir(low_res_dir) if f.endswith(".png")])
high_res_files = sorted([f for f in os.listdir(high_res_dir) if f.endswith(".png")])

# Compute ORB features for all images
low_res_features = {f: compute_features(os.path.join(low_res_dir, f)) for f in low_res_files}
high_res_features = {f: compute_features(os.path.join(high_res_dir, f)) for f in high_res_files}

# BFMatcher to find best matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Matching and copying process
for low_res in low_res_files:
    low_res_path = os.path.join(low_res_dir, low_res)
    low_desc = low_res_features[low_res]

    best_match = None
    best_score = float("inf")

    # Step 1: ORB Matching
    for high_res in high_res_files:
        high_res_path = os.path.join(high_res_dir, high_res)
        high_desc = high_res_features[high_res]

        if low_desc is not None and high_desc is not None:
            matches = bf.match(low_desc, high_desc)
            score = sum(m.distance for m in matches) / len(matches) if matches else float("inf")

            if score < best_score:
                best_score = score
                best_match = high_res

    # Step 2: Backup Matching (if no good ORB match found)
    if best_match is None or best_score > 100:  # If match is poor, use histogram backup
        best_match = max(high_res_files, key=lambda f: compute_histogram_similarity(low_res_path, os.path.join(high_res_dir, f)))

    # Copy the best-matching high-res tile
    if best_match:
        original_high_res_path = os.path.join(high_res_dir, best_match)
        new_high_res_path = os.path.join(output_dir, low_res)  # Copy with low-res filename

        shutil.copy(original_high_res_path, new_high_res_path)  # Copy instead of rename
        print(f"✅ Copied {best_match} → {new_high_res_path}")

print("\n🎉 Automatic tile copying complete! Check 'data/high_res' for results.")
