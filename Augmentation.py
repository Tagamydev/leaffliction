#!/usr/bin/env python3
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from srcs.Augmentation.transform_image import transform_image

# Supported image extensions
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

def collect_images(path):
    """
    Collect all image file paths from a single file or recursively from a directory.
    """
    image_files = []
    if os.path.isfile(path):
        if path.lower().endswith(IMG_EXTENSIONS):
            image_files.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.lower().endswith(IMG_EXTENSIONS):
                    image_files.append(file_path)
    else:
        print(f"Error: Path '{path}' does not exist.")
    return image_files

def process_images(image_files, max_workers=8):
    """
    Apply transformations to images concurrently using threads.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(transform_image, img): img for img in image_files}

        for future in as_completed(futures):
            img = futures[future]
            try:
                future.result()
                print(f"✅ Done: {img}")
            except Exception as e:
                print(f"❌ Failed: {img} -> {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: ./batch_transform.py <image_or_directory_path>")
        return

    path = sys.argv[1]
    image_files = collect_images(path)

    if not image_files:
        print("No images found.")
        return

    print(f"Found {len(image_files)} images. Processing with threads...")
    process_images(image_files, max_workers=8)  # adjust max_workers as needed

if __name__ == "__main__":
    main()

