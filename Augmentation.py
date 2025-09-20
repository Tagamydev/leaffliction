#!/usr/bin/env python3
import sys
import os
from srcs.Augmentation.transform_image import transform_image

# Supported image extensions
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

def process_path(path):
    """
    Process a single image or recursively traverse directories.
    """
    if os.path.isfile(path):
        # Single image
        if path.lower().endswith(IMG_EXTENSIONS):
            transform_image(path)
        else:
            print(f"Skipped non-image file: {path}")
    elif os.path.isdir(path):
        # Walk through all subdirectories
        for root, dirs, files in os.walk(path):
            # Sort files so that processing is predictable
            files.sort()
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.lower().endswith(IMG_EXTENSIONS):
                    transform_image(file_path)
    else:
        print(f"Error: Path '{path}' does not exist.")

def main():
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <image_or_directory_path>")
        return

    path = sys.argv[1]
    process_path(path)

if __name__ == "__main__":
    main()
