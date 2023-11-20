"""
    docstring
"""

import os
import uuid
import shutil
import cv2
import numpy as np
from PIL import Image

# SET BRIGHTNESS CONFIG HERE
MIN_BRIGHTNESS_THRESHOLD = 75
MAX_BRIGHTNESS_THRESHOLD = 175

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
    """
        auto brightness & auto sharpness
    """

    path_image = image
    splitted_path = path_image.split("/")
    
    modified_image = cv2.imread(image)
    # cv2.imwrite(str(splitted_path[0])+"/before"+str(splitted_path[1]),modified_image)
    gray_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
    average_brighness = np.mean(gray_image)
    
    print("Original Brightness : ", average_brighness)

    if average_brighness < MIN_BRIGHTNESS_THRESHOLD or average_brighness > MAX_BRIGHTNESS_THRESHOLD:
       modified_image = sharpen_image(modified_image, level=2) # Sharpen image to LV2, if image is Dark
       brightness_factor = MIN_BRIGHTNESS_THRESHOLD / average_brighness
       modified_image =  np.clip(modified_image * brightness_factor, 0, 255).astype(np.uint8)
       gray_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
       average_brighness = np.mean(gray_image)
       print("Adjusted Brightness : ", average_brighness)
       cv2.imwrite(str(splitted_path[0])+"/"+str(splitted_path[1]), modified_image)
    #    cv2.imwrite(str(splitted_path[0])+"/brightness"+str(splitted_path[1]), modified_image)
    else: 
        modified_image = sharpen_image(modified_image, level=1)  # Sharpen image to LV1, if image has normal brightness
        cv2.imwrite(str(splitted_path[0])+"/"+str(splitted_path[1]), modified_image)
        print("Image Brightness is Normal : ", average_brighness)

def sharpen_image(blur_image, level):
    # blur_image = cv2.imread(blur_image)
    # gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)

    if (level == 1):
        kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
        ])
        print("SHARPENED LV1")
    else:
        kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
        ])
        print("SHARPENED LV2")

    sharpened_image = cv2.filter2D(blur_image, -1, kernel)
    return sharpened_image