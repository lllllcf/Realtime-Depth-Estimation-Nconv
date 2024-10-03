import numpy as np
import cv2
import os

# Directory containing the .npy files
input_dir = './rawdepth'
output_dir = './coloredRawDepth'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        # Load the depth data from a .npy file
        file_path = os.path.join(input_dir, filename)
        raw_depth_data = np.load(file_path).reshape(480, 640)

        # Normalize the depth data
        raw_depth_data_normalized = cv2.normalize(raw_depth_data, None, 0, 255, cv2.NORM_MINMAX)
        raw_depth_data_8bit = raw_depth_data_normalized.astype(np.uint8)

        # Apply a color map
        colored_raw_depth = cv2.applyColorMap(raw_depth_data_8bit, cv2.COLORMAP_INFERNO)
        colored_raw_depth = cv2.rotate(colored_raw_depth, cv2.ROTATE_90_CLOCKWISE)

        # Save the colored depth data to a PNG file
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_file_path, colored_raw_depth)

print("All files have been processed and saved.")