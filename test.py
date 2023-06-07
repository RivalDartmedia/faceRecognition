import shutil
from deepface import DeepFace


temp_file_path = f"tmp/1.jpg"

# Process the uploaded file with DeepFace
result = DeepFace.verify(temp_file_path, temp_file_path)
print(result)