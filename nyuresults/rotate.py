from PIL import Image
import os

# Specify the directory containing the images
directory = './res0703/img'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
        img_path = os.path.join(directory, filename)
        with Image.open(img_path) as img:
            # Rotate the image
            rotated_img = img.rotate(-90, expand=True)
            # Save the rotated image, overwriting the original
            rotated_img.save(img_path)
            print(f"Rotated: {filename}")
    else:
        continue