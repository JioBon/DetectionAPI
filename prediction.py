from io import BytesIO

import numpy as np
from PIL import Image
import torch
import cv2


input_shape = (640, 640)
label = None

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


def load_onion_model():
    onion_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Onion.pt')
    print("Model loaded")
    return onion_model

onion_model = load_onion_model()

def predict(image: np.array, crop: str):
    global onion_model
    global onion_label_dict
    global label
    score = 0.00
    to_return = []

    if onion_model is None:
        print("loading")
        onion_model = load_onion_model()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = onion_model(image)
    print(results)

    detections = results.xyxy[0]
    print(detections)
    for detection in detections:
        label = detection[-1]
        label = onion_label_dict[int(label)]
        conf = detection[-2]
        score = float(conf)
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        to_return.append({"crop": crop, 'stress': label, 'score': f'{score:.2f}'})
    if not to_return:
        return [{"crop": crop, 'stress': 'HEALTHY', 'score': '1.00'}]
    else:
        return to_return

def read_imagefile(file) -> Image.Image:
    nparr = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def preprocess_img(image: Image.Image):
    # image = image.resize(input_shape)
    # image = np.array(image)
    
    # image = image / 127.5 - 1.0
    # image = np.expand_dims(image, 0)

    return image
