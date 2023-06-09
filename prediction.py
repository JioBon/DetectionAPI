from io import BytesIO

import numpy as np
from PIL import Image
import torch
import cv2
import tensorflow as tf
import sys

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
    corn_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Corn.pt')
    # corn_model = torch.load(sys.path.append('/custom_model/Corn.pt'))
    print("Corn Model loaded")
    return corn_model

def load_corn_model2():
    corn_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/backupv2/Corn.pt')
    # corn_model = torch.load(sys.path.append('/custom_model/Corn.pt'))
    print("Corn Model loaded")
    return corn_model

def load_eggplant_model():
    eggplant_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Eggplant.pt')
    # eggplant_model = torch.load(sys.path.append('/custom_model/Eggplant.pt'))
    print("Eggplant Model loaded")
    return eggplant_model

def load_eggplant_model2():
    eggplant_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/version2/Eggplant.pt')
    # eggplant_model = torch.load(sys.path.append('/custom_model/Eggplant.pt'))
    print("Eggplant Model loaded")
    return eggplant_model

def load_onion_model():
    onion_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Onion.pt')
    # onion_model = torch.load(sys.path.append('/custom_model/Onion.pt'))
    print("Onion Model loaded")
    return onion_model

def load_onion_model2():
    onion_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/version2/Onion.pt')
    # onion_model = torch.load(sys.path.append('/custom_model/Onion.pt'))
    print("Onion Model loaded")
    return onion_model

def load_tomato_model():
    tomato_model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Tomato.pt')
    # tomato_model = torch.load(sys.path.append('/custom_model/Tomato.pt'))
    print("Tomato Model loaded")
    return tomato_model

def load_tomato_model2():
    tomato_model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/version2/Tomato.pt')
    # tomato_model = torch.load(sys.path.append('/custom_model/Tomato.pt'))
    print("Tomato Model loaded")
    return tomato_model

def load_image_detect():
    ImageDetect_model = tf.keras.models.load_model("custom_model/saganaImageDetection.h5")
    print("Image Detection Model loaded")
    return ImageDetect_model

corn_model = load_corn_model()
corn_model2 = load_corn_model2()

eggplant_model = load_eggplant_model()
eggplant_model2 = load_eggplant_model2()

onion_model = load_onion_model()
onion_model2 = load_onion_model2()

tomato_model = load_tomato_model()
tomato_model2 = load_tomato_model2()
ImageDetect_model = load_image_detect()

crop_classes = ['corn', 'eggplant', 'noncrop', 'onion', 'tomato']

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
        results2 = corn_model2(image)
    elif crop.lower() == "eggplant":
        results = eggplant_model(image)
        results2 = eggplant_model2(image)
    elif crop.lower() == "onion":
        results = onion_model(image)
        # if results:
        results2 = onion_model2(image)
    # elif crop.lower() == "tomato":
    else:
        results = tomato_model(image)
        results2 = tomato_model2(image)
    
    print("hello", type(results))

    detections = results.xyxy[0]
    detections2 = results2.xyxy[0]
    print("testing", type(detections))
    to_return = filterDetection(detections, crop, image)
    check = filterDetection(detections2, crop, image)
    if check:
        to_return.extend(check)  
    if not to_return:
        return [
            {"crop": crop, 'stress': 'HEALTHY', 'score': '1.00', 
            'x1': f'0', 'y1': f'0', 'x2': f'0', 'y2': f'0'}
            ]
    else:
        return to_return

def filterDetection(detections, crop, image):
    to_return = []
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
        print(f"{label} with {score}")

        # print(label, score)
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if score >= 0.2:
            to_return.append(
                {"crop": crop, 'stress': label, 'score': f'{score:.2f}', 
                 'x1': f'{x1}', 'y1': f'{y1}', 'x2': f'{x2}', 'y2': f'{y2}'}
                )

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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = tf.image.resize(image, (224, 224))
    input_tensor = tf.expand_dims(resized_image, 0)

    prediction = ImageDetect_model.predict(input_tensor)
    prediction = tf.nn.softmax(prediction)
    predicted_label = np.argmax(prediction[0])

    return crop_classes[predicted_label]
