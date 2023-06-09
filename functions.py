"""
    docstring
"""

import os
import uuid
import shutil
from PIL import Image

def resize_image_dimension(file_path):
    """
        docstring
    """

    image = Image.open(file_path)
    width, height = image.size
    width = round(width*800/height)
    new_image = image.resize((width, 800))
    new_image.save(file_path)

    return file_path

def file_to_image(file):
    """
        docstring
    """

    unique = str(uuid.uuid4())[:8]
    extension = os.path.splitext(file.filename)[1]
    file_path   = f"tmp/{unique}{extension}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path
