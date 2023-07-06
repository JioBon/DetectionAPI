from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io
from PIL import Image
import torch
import numpy as np
import cv2
from prediction import read_imagefile, predict, check_image
from tensorflow import keras

app = FastAPI()
uploaded_files = []

class Msg(BaseModel):
    msg: str

@app.post("/baseimg/{file.filename}")
async def create_upload_file(file: UploadFile = File(...), crop: str = Form(...)):
    return {"filename": file.filename, "crop": crop}

@app.route("/baseimg/{filename}", methods=["GET"])
async def check_upload_file(filename: str):
    return FileResponse(filename)

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), crop: str = Form(...), bypass: bool = Form(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    # image = preprocess_img(image)
    crop_type = check_image(image)
    print(crop_type, crop)
    if bypass:
        prediction = predict(image, crop)
        return prediction
    elif crop_type == 'noncrop':
        return "Not a Crop"
    else:
        if crop_type.lower() == crop.lower():
            prediction = predict(image, crop)
            return prediction
        else:
            return f'Not {crop}'

        


# ==============================================================================================================
# @app.get("/")
# async def root():
#     return {"message": "Hello World. Welcome to FastAPI!"}


# @app.get("/path")
# async def demo_get():
#     return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


# @app.post("/path")
# async def demo_post(inp: Msg):
#     return {"message": inp.msg.upper()}


# @app.get("/path/{path_id}")
# async def demo_get_path_id(path_id: int):
#     return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}