from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io
from PIL import Image
import torch
import numpy as np
import cv2
from prediction import read_imagefile, predict, check_image


# DICTS
onion_label_dict = {
    0: 'dog',
    1: 'person',
    2: 'cat',
    3: 'tv',
    4: 'car',
    5: 'meatballs',
    6: 'marinara sauce',
    7: 'tomato soup',
    8: 'chicken noodle soup',
    9: 'french onion soup',
    10: 'chicken breast',
    11: 'ribs',
    12: 'pulled pork',
    13: 'hamburger',
    14: 'cavity',
    15: 'leaf miners',
    16: 'downy mildew',
    17: 'botrytis leaf blight',
    18: 'armyworm'
}

# MODELS
onion_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Onion.pt')

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
async def predict_api(file: UploadFile = File(...), crop: str = Form(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    # image = preprocess_img(image)
    if check_image(image):
        return [{"crop": 'Not Crop', 'stress': 'Not a Crop', 'score': '1.00', 
            'x1': f'0', 'y1': f'0', 'x2': f'0', 'y2': f'0'}]
    else:
        prediction = predict(image, crop)

        return prediction


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