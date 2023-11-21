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
from functions import*
from pydantic import BaseModel
from datetime import datetime
from fastapi import Request

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

DETECTORS = [
    "opencv", 
    "retinaface", 
    "mtcnn", 
    "ssd", 
    "dlib", 
    "mediapipe",
]

# SET FACE-RECOG CONFIG HERE
USED_MODEL = MODELS[2]
USED_DETECTOR = DETECTORS[2]
VALID_DISTANCE = 0.37
apiKey = "bGjJKURn3HPeafvE/BRv2MMfe3F6VRpf9qUbv4Q6Qf4="

class ResultResponse(BaseModel):
        verified:bool = False
        distance:float = VALID_DISTANCE
class VerifyResponse(BaseModel):
    status: int = 200
    result: ResultResponse
    
async def log_request(request: Request, call_next):
    # Cetak detail permintaan HTTP
    print(f"Time: {datetime.now()}\t")
    print(f"Received request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    print(f"Query Parameters: {request.query_params}")
    print(f"Path Parameters: {request.path_params}")
    # print(f"Body: {await request.body()}")

    # Panggil fungsi berikutnya dalam rantai middleware atau aplikasi utama
    response = await call_next(request)

    return response

app.middleware('http')(log_request)

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
        print(f"{datetime.now()}\t[Error Authorization]")
        print(e)
        raise HTTPException(status_code=403, detail="Invalid Authorization header")
    provided_api_key = authorization.split(" ")[1]
    expected_api_key = "bGjJKURn3HPeafvE/BRv2MMfe3F6VRpf9qUbv4Q6Qf4="
    if provided_api_key != expected_api_key:
        print(f"{datetime.now()}\t[Error Invalid API Key]")
        print(e)
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

        check_brightness(image1_path)
        check_brightness(image2_path)

        # Process the uploaded file with DeepFace
        result = DeepFace.verify(
            image1_path,
            image2_path,
            enforce_detection=False,
            model_name=USED_MODEL,
            detector_backend=USED_DETECTOR
        )

        result['verified'] = bool(result['verified'].item())
        distance = result['distance']
        verified = result['verified']
        rounded_distance = round(distance, 2)

        if (rounded_distance <= VALID_DISTANCE or verified == True):
            verified = True
        else:
            verified = False

        return JSONResponse(content={
            "status":200,
            "result": {
                "verified":verified,
                "distance":rounded_distance,
            }})

    except Exception as e:
        print(f"{datetime.now()}\t[Error Exception]")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)


# Post request for testing
@app.post("/api/facematch/v1/verify",response_model=VerifyResponse)
async def create_upload_file(file1: UploadFile = File(...),file2: UploadFile = File(...)):
    """
        Example Success Response
        {
            "status": 200,
            "result": {
                "verified": false,
                "distance": 0.36
            }
        }
    """
    try:
        #  Save the uploaded file to a temporary directory on disk
        file_path1 = file_to_image(file1) 
        file_path2 = file_to_image(file2)

        _ = resize_image_dimension(file_path1)
        _ = resize_image_dimension(file_path2)


        check_brightness(file_path1)
        check_brightness(file_path2)

        # Process the uploaded file with DeepFace
        result = DeepFace.verify(
            file_path1,
            file_path2,
            enforce_detection=False,
            model_name=USED_MODEL,
            detector_backend=USED_DETECTOR
        )

        result['verified'] = bool(result['verified'].item())
        distance = result['distance']
        verified = result['verified']
        rounded_distance = round(distance, 2)

        if (rounded_distance <= VALID_DISTANCE or verified == True):
            verified = True
        else:
            verified = False

        return JSONResponse(content={
            "status":200,
            "result": {
                "verified":verified,
                "distance":rounded_distance,
            }})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path1):
            os.remove(file_path1)
        if os.path.exists(file_path2):
            os.remove(file_path2)


if __name__ == "__main__":
    # uvicorn.run(app, host="119.10.176.108", port=9001)
    uvicorn.run(app, host="localhost", port=9005)
