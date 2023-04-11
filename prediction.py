from io import BytesIO

import numpy as np
from PIL import Image
import torch
import cv2
import tensorflow as tf

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

corn_label_dict = {
    0: "dog",
    1: "person",
    2: "cat",
    3: "tv",
    4: "car",
    5: "meatballs",
    6: "marinara sauce",
    7: "tomato soup",
    8: "chicken noodle soup",
    9: "french onion soup",
    10: "chicken breast",
    11: "ribs",
    12: "pulled pork",
    13: "hamburger",
    14: "cavity",
    15: "leaf spot",
    16: "corn borer",
    17: "eyes spot",
    18: "goss's wilt",
    19: "powdery mildew",
    20: "armyworm",
    21: "corn plant hopper",
    22: "Corn borer midrib feeding",
    23: "adult armyworm",
    24: "fall armyworm eggs"
}

tomato_label_dict = {
    0: "Leafminer",
    1: "Fusarium Wilt",
    2: "Black Mold",
    3: "Powdery Mildew"
}

eggplant_label_dict = {
    0: "Leaf Miners",
    1: "White Flies",
    2: "Powdery Mildew",
    3: "Flea Beetles",
    4: "Holes caused by Aphids",
    5: "Flea Beetle's damage",
    6: "Leaf Spot",
    7: "Potato Beetle",
    8: "Aphids",
    9: "Armyworm",
    10: "Leaf roller moth",
}

def load_corn_model():
    corn_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Corn.pt')
    print("Corn Model loaded")
    return corn_model

def load_eggplant_model():
    eggplant_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Eggplant.pt')
    print("Eggplant Model loaded")
    return eggplant_model

def load_onion_model():
    onion_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Onion.pt')
    print("Onion Model loaded")
    return onion_model

def load_tomato_model():
    tomato_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Tomato.pt')
    print("Tomato Model loaded")
    return tomato_model

def load_image_detect():
    ImageDetect_model = tf.keras.models.load_model("custom_model/saganaImageDetection.h5")
    print("Image Detection Model loaded")
    return ImageDetect_model

corn_model = load_corn_model()
eggplant_model = load_eggplant_model()
onion_model = load_onion_model()
tomato_model = load_tomato_model()
ImageDetect_model = load_image_detect()

def predict(image: np.array, crop: str):
    global onion_model
    global corn_model
    global eggplant_model
    global tomato_model
    global onion_label_dict
    global label
    score = 0.00
    to_return = []

    # if corn_model is None:
    #     print("loading")
    #     corn_model = load_corn_model()

    # if eggplant_model is None:
    #     print("loading")
    #     eggplant_model = load_eggplant_model()

    # if onion_model is None:
    #     print("loading")
    #     onion_model = load_onion_model()

    # if tomato_model is None:
    #     print("loading")
    #     tomato_model = load_tomato_model()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop.lower() == "corn":
        results = corn_model(image)
    elif crop.lower() == "eggplant":
        results = eggplant_model(image)
    elif crop.lower() == "onion":
        results = onion_model(image)
    # elif crop.lower() == "tomato":
    else:
        results = tomato_model(image)
    
    print(results)

    detections = results.xyxy[0]
    print(detections)
    for detection in detections:
        label = detection[-1]
        if crop.lower() == "corn":
            label = corn_label_dict[int(label)]
        elif crop.lower() == "eggplant":
            label = eggplant_label_dict[int(label)]
        elif crop.lower() == "onion":
            label = onion_label_dict[int(label)]
        # elif crop.lower() == "tomato":
        else:
           label = tomato_label_dict[int(label)]
        
        conf = detection[-2]
        score = float(conf)
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if score >= 0.4:
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

def check_image(image: np.array):
    global ImageDetect_model
    class_names = ['crop', 'noncrop']

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = tf.image.resize(image, (224, 224))
    input_tensor = tf.expand_dims(resized_image, 0)

    prediction = ImageDetect_model.predict(input_tensor)
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5, 0, 1)

    return prediction.numpy()[0][0]
