import os
from deepface import DeepFace

db_path = "my_db"
similarity_metric = "cosine"
threshold = 0.27  # Define the threshold for cosine similarity

# Perform the face recognition search
results = DeepFace.find(img_path="./test.jpeg", 
                        db_path=db_path, 
                        model_name="Facenet512",
                        distance_metric=similarity_metric,
                        enforce_detection=False)

# Check if any matches were found
if results:
    # Iterate over each DataFrame in the results
    for df in results:
        for index, row in df.iterrows():
            file_path = row["identity"]
            file_name = os.path.basename(file_path)
            cosine_similarity = row.get(similarity_metric, None)
            print(f"file_path:{file_path}\n{cosine_similarity}")
            if cosine_similarity is not None:
                is_same_person = cosine_similarity < threshold
                print(f"Match found at: filename: {file_path}, cosine_similarity: {cosine_similarity}, same person: {is_same_person}")
            else:
                print(f"Match found at: filename: {file_path}, but no similarity score found.")
else:
    print("No matches found.")