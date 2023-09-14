"""
    docstring
"""
# pylint: disable=import-error

import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from functions import resize_image_dimension, file_to_image, check_brightness
from pydantic import BaseModel


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# penggunaan GPU
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
]
class ResultResponse(BaseModel):
        verified:bool = False
        distance:float = 0.36
class VerifyResponse(BaseModel):
    status: int = 200
    result: ResultResponse

apiKey = "bGjJKURn3HPeafvE/BRv2MMfe3F6VRpf9qUbv4Q6Qf4="

@app.get("/api/facematch/v1/verify",response_model=VerifyResponse)
async def create_upload_file(file1:str,file2:str,authorization:str = Header(...,description="API key for authentication using Bearer")):
    """
        Success Response
        {
            "status": 200,
            "result": {
                "verified": false,
                "distance": 0.36
            }
        }
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid Authorization header")
    provided_api_key = authorization.split(" ")[1]
    expected_api_key = "bGjJKURn3HPeafvE/BRv2MMfe3F6VRpf9qUbv4Q6Qf4="
    if provided_api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    try:
        image1_path = "tmp/image1.jpg"
        image2_path = "tmp/image2.jpg"
        with open(image1_path, "wb") as img_file:
            img_file.write(requests.get(file1).content)
        with open(image2_path, "wb") as img_file:
            img_file.write(requests.get(file2).content)

        _ = resize_image_dimension(image1_path)
        _ = resize_image_dimension(image2_path)

        image1_path = check_brightness(image1_path)
        image2_path = check_brightness(image2_path)

        # Process the uploaded file with DeepFace
        result = DeepFace.verify(
            image1_path,
            image2_path,
            enforce_detection=False,
            model_name=MODELS[2],
            detector_backend='mtcnn'
        )

        result['verified'] = bool(result['verified'].item())
        distance = result['distance']
        verified = result['verified']
        rounded_distance = round(distance, 2)
        return JSONResponse(content={
            "status":200,
            "result": {
                "verified":verified,
                "distance":rounded_distance,
            }})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)
        
# @app.post("/api/facematch/v1/verify")
# async def create_upload_file(file1: UploadFile = File(...),file2: UploadFile = File(...)):
#     """
#         Success Response
#         {
#             "status": 200,
#             "result": {
#                 "verified": false,
#                 "distance": 0.36
#             }
#         }
#     """
#     # pylint: disable=raise-missing-from,invalid-name

#     if file1.filename == '' or file2.filename == '':
#         return JSONResponse(status_code=400,content={
#             "status":400,
#             "message":"file1 and file2 are required"
#         })
#     try:
#         # Save the uploaded file to a temporary directory on disk
#         file_path1 = file_to_image(file1)
#         file_path2 = file_to_image(file2)

#         _ = resize_image_dimension(file_path1)
#         _ = resize_image_dimension(file_path2)

#         # Process the uploaded file with DeepFace
#         result = DeepFace.verify(
#             file_path1,
#             file_path2,
#             enforce_detection=False,
#             model_name=MODELS[2]
#         )

#         result['verified'] = bool(result['verified'].item())
#         distance = result['distance']
#         verified = result['verified']
#         rounded_distance = round(distance, 2)
#         # Return a JSON response indicating success
#         return JSONResponse(content={
#             "status":200,
#             "result": {
#                 "verified":verified,
#                 "distance":rounded_distance,
#             }})

#     except Exception as e:
#         # If an error occurs, raise an HTTPException with a 500 status code
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         # Always close and remove the temporary file when finished
#         file1.file.close()
#         file2.file.close()
#         os.remove(file_path1)
#         os.remove(file_path2)


if __name__ == "__main__":
    uvicorn.run(app, host="119.10.176.108", port=9001)
