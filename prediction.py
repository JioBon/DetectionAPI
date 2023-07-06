from io import BytesIO

import numpy as np
from PIL import Image
import torch
import cv2
import tensorflow as tf
import sys
from tensorflow import keras

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
    15: 'Leaf Miners Damage',
    16: 'Armyworm Damage',
    17: 'Botrytis Leaf Blight',
    18: 'Beet Armyworm'
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
    15: "Scraping Damage of Armyworm",
    16: "Corn Borer Damage",
    17: "Eye Spot",
    18: "Goss's Wilt",
    19: "Corn Plant Hopper Egg Mass",
    20: "Fall Armyworm Larve",
    21: "Corn Plant Hopper",
    22: "Frass of Armyworm",
    23: "Fall armyworm Female Moth",
    24: "Fall Armyworm Egg Mass"
}

tomato_label_dict = {
    0: "Leaf Miner",
    1: "Fusarium Wilt",
    2: "Black Mold",
    3: "Powdery Mildew"
}

eggplant_label_dict = {
    0: "Leaf Miner",
    1: "Leaf Hopper",
    2: "Powdery Mildew",
    3: "Flea Beetles",
    4: "Holes caused by Aphids",
    5: "Flea Beetle's Damage",
    6: "Leaf Spot",
    7: "Potato Beetle",
    8: "Aphids",
    9: "Earworm",
    10: "Leaf Roller Moth",
}

def load_corn_model():
    # corn_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Corn.pt')
    # corn_model = torch.load(sys.path.append('/custom_model/Corn.pt'))

    corn_model = tf.keras.models.load_model("custom_model/Corn.h5")
    print("Corn Model loaded")
    return corn_model

def load_eggplant_model():
    # eggplant_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Eggplant.pt')
    # eggplant_model = torch.load(sys.path.append('/custom_model/Eggplant.pt'))

    eggplant_model = tf.keras.models.load_model("custom_model/Eggplant.h5")
    print("Eggplant Model loaded")
    return eggplant_model

def load_onion_model():
    # onion_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Onion.pt')
    # onion_model = torch.load(sys.path.append('/custom_model/Onion.pt'))

    onion_model = tf.keras.models.load_model("custom_model/Onion.h5")
    print("Onion Model loaded")
    return onion_model

def load_tomato_model():
    # tomato_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Tomato.pt')
    # tomato_model = torch.load(sys.path.append('/custom_model/Tomato.pt'))

    tomato_model = tf.keras.models.load_model("custom_model/Tomato.h5")
    print("Tomato Model loaded")
    return tomato_model

def load_image_detect():
    ImageDetect_model = tf.keras.models.load_model("custom_model/saganaImageDetection.h5")
    print("Image Detection Model loaded")
    return ImageDetect_model

ImageDetect_model = load_image_detect()
corn_model = load_corn_model()
eggplant_model = load_eggplant_model()
onion_model = load_onion_model()
tomato_model = load_tomato_model()


crop_classes = ['corn', 'eggplant', 'noncrop', 'onion', 'tomato']

corn_classes = ['Corn Borer Damage', 'Corn Plant Hopper', 'Goss_s Wilt', 'Scraping Damage of Armyworms']
eggplant_classes = ['Earworm', "Flea Beetle's damage", 'Leaf Spot']
onion_classes = ['armyworm damage', 'beet armyworm', 'botrytis leaf blight', 'leaf miners damage']
tomato_classes = ['Black Mold', 'Fusarium Wilt', "Leaf Miner's damage"]

def predict(image: np.array, crop: str):
    global onion_model
    global corn_model
    global eggplant_model
    global tomato_model

    global corn_classes
    global eggplant_classes
    global onion_classes
    global tomato_classes

    global label
    score = 0.00
    to_return = []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = tf.image.resize(image, (224, 224))
    input_tensor = tf.expand_dims(resized_image, 0)


    # prediction = ImageDetect_model.predict(input_tensor)
    if crop.lower() == "corn":
        prediction = corn_model.predict(input_tensor)
        results = get_label(prediction)
    elif crop.lower() == "eggplant":
        prediction = eggplant_model.predict(input_tensor)
        results = get_label(prediction)
    elif crop.lower() == "onion":
        prediction = onion_model.predict(input_tensor)
        results = get_label(prediction)
    # elif crop.lower() == "tomato":
    else:
        prediction = tomato_model.predict(input_tensor)
        results = get_label(prediction)

    prediction = tf.nn.softmax(prediction)
    predicted_label = np.argmax(prediction[0])
    
    print(results)
   
    if not results:
        return [
            {"crop": crop, 'stress': 'HEALTHY', 'score': '1.00', 
            'x1': f'0', 'y1': f'0', 'x2': f'0', 'y2': f'0'}
            ]
    else:
        return results
    
def get_label(prediction, crop):
    prediction = tf.nn.softmax(prediction)
    predicted_label = np.argmax(prediction[0])

    if crop.lower() == "corn":
        return corn_classes[predicted_label]
    elif crop.lower() == "eggplant":
        return eggplant_classes[predicted_label]
    elif crop.lower() == "onion":
        return onion_classes[predicted_label]
    # elif crop.lower() == "tomato":
    else:
        return tomato_classes[predicted_label]

    

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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = tf.image.resize(image, (224, 224))
    input_tensor = tf.expand_dims(resized_image, 0)

    prediction = ImageDetect_model.predict(input_tensor)
    prediction = tf.nn.softmax(prediction)
    predicted_label = np.argmax(prediction[0])

    return crop_classes[predicted_label]
