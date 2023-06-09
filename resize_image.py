from PIL import Image

def resize_image_dimension(file_path):

    image = Image.open(file_path)
    width, height = image.size
    width = round(width*800/height)
    new_image = image.resize((width, 800))
    new_image.save(file_path)

    return file_path