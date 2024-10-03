import os
from PIL import Image, ImageDraw, ImageFont

def merge_images(base_dir):
    # List all directories in the base directory
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Check if the number of directories is exactly eight
    if len(dirs) != 8:
        raise ValueError("There must be exactly eight directories in the base directory.")

    # Create a dictionary to store images by their names
    images_by_name = {}

    # Process each directory
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            if img_name.endswith('.png'):
                if img_name not in images_by_name:
                    images_by_name[img_name] = []
                image = Image.open(img_path)
                draw = ImageDraw.Draw(image)
                # Use a larger font size with the default font
                font = ImageFont.load_default()
                draw.text((10, 10), dir_name, fill="white", font=font)
                images_by_name[img_name].append(image)

    # Merge images with the same name and save them
    for img_name, images in images_by_name.items():
        if len(images) != 8:
            raise ValueError(f"Image '{img_name}' does not have exactly 8 versions across directories.")

        # Determine the maximum width and maximum height of the individual images
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create a new image with 2 rows and 4 columns
        new_img = Image.new('RGB', (max_width * 4, max_height * 2))

        # Paste each image into the new image
        for index, img in enumerate(images):
            row = index // 4
            col = index % 4
            new_img.paste(img, (col * max_width, row * max_height))

        # Save the merged image
        new_img.save(os.path.join(base_dir, img_name))

if __name__ == "__main__":
    base_dir = './res0703'
    merge_images(base_dir)