"""
    docstring
"""
# pylint: disable=import-error
import json
import uuid
import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
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
class ResultResponseFind(BaseModel):
        code: int
        status: str
        name: str
class VerifyResponseFind(BaseModel):
    status: int = 200
    result: ResultResponseFind
    
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
        random_uuid1 = uuid.uuid4()
        image1_path = os.path.join("tmp", str(random_uuid1) + ".jpg")
        random_uuid2 = uuid.uuid4()
        image2_path = os.path.join("tmp", str(random_uuid2) + ".jpg")
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
            

@app.post("/api/facematch/v1/register", response_model=VerifyResponseFind)
async def register_user(image: UploadFile = File(...), name: str = Form(...)):
    try:
        # Save the uploaded file to the database directory
        image_path = save_uploaded_file(image)
        
        # Load the existing JSON database
        data_db = load_json_db()

        # Append new user data to the JSON database
        data_db.append({"path": image_path, "nama": name})
        save_json_db(data_db)

        # Delete the existing representations file if it exists
        if os.path.exists(representation_file):
            os.remove(representation_file)

        # Regenerate the representations file using DeepFace
        DeepFace.find(
            img_path=image_path, 
            db_path=db_path, 
            model_name="Facenet512",
            distance_metric="cosine",
            enforce_detection=False
        )

        return JSONResponse(content={
            "status": 200,
            "result": {
                "code": 0,
                "status": "registered",
                "name": name
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
            
@app.post("/api/facematch/v1/find", response_model=VerifyResponseFind)
async def create_upload_file(image: UploadFile = File(...)):
    """
        Example Success Response
        {
            "status": 200,
            "result": {
                "code": 0,
                "status": "match",
                "name": "John Doe"
            }
        }
    """
    image_path = None
    try:
        threshold = 0.3
        db_path = "my_db"
        similarity_metric = "cosine"
        
        # Save the uploaded file to a temporary directory on disk
        image_path = file_to_image(image)

        # Placeholder brightness check function
        check_brightness(image_path)
        
        # Load the JSON data
        with open("data_db.json", "r") as f:
            data_db = json.load(f)

        # Convert list of dictionaries to a dictionary for quick lookup
        path_to_name = {entry["path"]: entry["nama"] for entry in data_db}

        # Process the uploaded file with DeepFace
        results = DeepFace.find(
            img_path=image_path, 
            db_path=db_path, 
            model_name="Facenet512",
            distance_metric=similarity_metric,
            enforce_detection=False
        )
        print(f"result: {results}")
        
        if results:
            # Since the results might contain multiple DataFrames, iterate over them
            df = results[0]
            if not df.empty:
                # Get the first row of the DataFrame
                row = df.iloc[0]
                file_path = row["identity"]
                cosine_similarity = row.get(f"Facenet512_{similarity_metric}", None)
                
                if cosine_similarity is not None and cosine_similarity < threshold:
                    print(f"file_path: {file_path}")
                    name = path_to_name.get(file_path, "Unknown")
                    return JSONResponse(content={
                        "status": 200,
                        "result": {
                            "code": 0,
                            "status": "match",
                            "name": name
                        }
                    })
        # if results:
        #     # Iterate over each DataFrame in the results
        #     for df in results:
        #         for index, row in df.iterrows():
        #             file_path = row["identity"]
        #             file_name = os.path.basename(file_path)
        #             cosine_similarity = row.get(similarity_metric, None)
        #             print(f"similarity: {row.get(similarity_metric, None)}")
        #             is_same_person = cosine_similarity < threshold
        #             print(f"similarity: {cosine_similarity} && filename: {file_path}, {is_same_person}")
        #             if is_same_person:
        #                 print(f"file_path:{file_path}\n{cosine_similarity}")
        #                 name = path_to_name.get(file_path, "Unknown")
        #                 return JSONResponse(content={
        #                     "status": 200,
        #                     "result": {
        #                         "code": 0,
        #                         "status": "match",
        #                         "name": name
        #                     }
        #                 })

        # If no match was found in the results
        return JSONResponse(content={
            "status": 200,
            "result": {
                "code": 20,
                "status": "not match",
                "name": "None"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    uvicorn.run(app, host="202.43.169.13", port=9001)
    # uvicorn.run(app, host="localhost", port=9001)
