"""
    docstring
"""

import os
import uuid
import shutil
import cv2
import numpy as np
from PIL import Image

MIN_BRIGHTNESS_THRESHOLD = 100
MAX_BRIGHTNESS_THRESHOLD = 150

def resize_image_dimension(file_path):
    """
        docstring
    """

    image = Image.open(file_path)
    width, height = image.size

    if width <= 800 or height <= 800:
        return file_path

    width = round(width*800/height)
    new_image = image.resize((width, 800))
    new_image.save(file_path)

    return file_path

def file_to_image(file):

    unique = str(uuid.uuid4())[:8]
    extension = os.path.splitext(file.filename)[1]
    file_path   = f"tmp/{unique}{extension}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path


def check_brightness(image):
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brighness = np.mean(gray_image)
    print("Original Brightness : ", average_brighness)

    if average_brighness < MIN_BRIGHTNESS_THRESHOLD or average_brighness > MAX_BRIGHTNESS_THRESHOLD:
       brightness_factor = MIN_BRIGHTNESS_THRESHOLD / average_brighness
       image =  np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
       gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       average_brighness = np.mean(gray_image)
       print("Adjusted Brightness : ", average_brighness)
       return image
    else:
        print("Image Brightness is Normal : ", average_brighness)
        return image