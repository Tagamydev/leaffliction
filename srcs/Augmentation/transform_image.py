#!/usr/bin/env python3
from PIL import Image, ImageOps
import os

def transform_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: file '{image_path}' does not exist.")
        return

    img = Image.open(image_path)
    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # --- 1. Flip ---
    flipped = ImageOps.mirror(img)
    flipped.save(os.path.join(dir_name, f"{base_name}_flip.jpg"))

    # --- 2. Rotate ---
    rotated = img.rotate(45)
    rotated.save(os.path.join(dir_name, f"{base_name}_rotate.jpg"))

    # --- 3. Crop ---
    width, height = img.size
    cropped = img.crop((width//4, height//4, 3*width//4, 3*height//4))
    cropped.save(os.path.join(dir_name, f"{base_name}_crop.jpg"))

    # --- 4. Shear / Skew ---
    m = -0.5  # shear factor
    xshift = abs(m) * height
    new_width = width + int(round(xshift))
    sheared = img.transform(
        (new_width, height),
        Image.AFFINE,
        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC
    )
    sheared.save(os.path.join(dir_name, f"{base_name}_shear.jpg"))

    # --- 5. Distortion ---
    coeffs = [1, 0.2, 0, 0.2, 1, 0]
    distorted = img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
    distorted.save(os.path.join(dir_name, f"{base_name}_distortion.jpg"))

    print(f"Transformations for '{image_path}' saved in '{dir_name}'")
