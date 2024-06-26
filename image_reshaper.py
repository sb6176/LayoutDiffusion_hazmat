import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Open an image file
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Calculate new size and aspect ratio
                width, height = img.size
                if width > height:
                    new_height = size
                    new_width = int((new_height / height) * width)
                else:
                    new_width = size
                    new_height = int((new_width / width) * height)

                # Resize the image while maintaining aspect ratio
                img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

                # Create a new blank image (white background)
                new_img = Image.new("RGB", (size, size), (255, 255, 255))

                # Center the resized image on the blank image
                offset_x = (size - new_width) // 2
                offset_y = (size - new_height) // 2
                new_img.paste(img_resized, (offset_x, offset_y))

                # Save the resized image in the output directory
                new_img.save(os.path.join(output_folder, filename))

    print(f"All images have been resized and saved to {output_folder}")

# Usage example
input_folder = 'images/images/trainval/'
output_folder = 'images/images256/trainval/'
square_size = 256  # Desired square size

resize_images(input_folder, output_folder, square_size)
