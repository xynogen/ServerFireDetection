import tensorflow as tf
import numpy as np
from firebase_admin import credentials, db, messaging, initialize_app, storage
from PIL import ImageFont, ImageDraw, Image
import cv2
import time


URL = 'https://firedetection-970d0-default-rtdb.asia-southeast1.firebasedatabase.app/'
CRED = credentials.Certificate("firedetection.json")
APP = initialize_app(CRED, {"storageBucket": "firedetection-970d0.appspot.com"})
IMAGE = "Ruangan.jpg"
IMAGE_PATH = f"tmp/{IMAGE}"
SAMPLE_VIDEO = "sample.mp4"

# Setting Path Model yang dipakai
MODEL_PATH = "ModelFireDetection/model1/"
model = tf.keras.models.load_model(MODEL_PATH)

# Setting mengenai Tampilan
CLASS = ["FIRE", "NON-FIRE"]
THRESHOLD = 0.5
DELTAT = 5
DISPLAY_ONLY = False
COLOR_NON_FIRE = "#348e40"
COLOR_FIRE = "#ed2f2f"
TEXT_COLOR = (255, 255, 255, 0)
FONT = ImageFont.truetype('arial.ttf', 24)
TEXTx = 20 
TEXTy = 20

# variable untuk menghitung FPS
prev_frame_time = 0
new_frame_time = 0
prev_send_time = 0
new_send_time = 0

# variable frame untuk pengiriman gambar
frame = 0

def send_image():
    ref = db.reference("/", APP, URL)
    ref.child("event").set("0")
    bucket = storage.bucket(app=APP)
    blob = bucket.blob(IMAGE)
    blob.upload_from_filename(IMAGE_PATH)

def send_notif():
    message = messaging.Message(
        notification=messaging.Notification(
            title='FireDetection',
            body='Ada Api Terdeteksi',
        ),
        condition="'notif' in topics",
    )
    response = messaging.send(message)

def check_event(frame):
    ref = db.reference("/", APP, URL)
    if ref.child("event").get() == "1":
        send_image()
        ref.child("event").set("0")

# buka kamera
cap = cv2.VideoCapture(SAMPLE_VIDEO)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# main server loop
frame_counter = 0
index = 0
skip = 50

while True:
    new_frame_time = time.time()
    new_send_time = new_frame_time
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, C = frame.shape
    
    if frame_counter % skip == 0:
        image = frame/255
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        prediction = round(float(prediction[0][0]), 3)

        index = 1
        if prediction < THRESHOLD:
            index = 0
            
        text = f"{CLASS[index]}: {prediction}"

    if CLASS[index] == "FIRE":
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
    
        # gambar kotak penanda
        # shape class
        shape = [(TEXTx, TEXTy), (TEXTx+150, TEXTy + 30)]
        draw.rectangle(shape, fill= COLOR_FIRE)
        # shape fps
        shape = [(TEXTx, TEXTy+34), (TEXTx+150, TEXTy + 60)]
        draw.rectangle(shape, fill= COLOR_FIRE)
        # shape border
        shape = [(0, 0), (W-1, H-1)]
        draw.rectangle(shape, outline= COLOR_FIRE, width=10)
        
        # calculate FPS
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = 'FPS: ' + str(fps)

        # show text
        draw.text((TEXTx, TEXTy + 34), fps, font = FONT, fill = TEXT_COLOR)
        draw.text((TEXTx, TEXTy), text, font = FONT, fill = TEXT_COLOR)
        
        frame = np.array(img_pil)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if prediction < THRESHOLD:
            if DISPLAY_ONLY != True:
                if new_send_time - prev_send_time > DELTAT:
                    cv2.imwrite(IMAGE_PATH, frame)
                    send_image()
                    send_notif()
                    prev_send_time = time.time()

    else:
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
    
        # gambar kotak penanda
        # shape class
        shape = [(TEXTx, TEXTy), (TEXTx+210, TEXTy + 30)]
        draw.rectangle(shape, fill= COLOR_NON_FIRE)
        # shape fps
        shape = [(TEXTx, TEXTy+34), (TEXTx+210, TEXTy + 60)]
        draw.rectangle(shape, fill= COLOR_NON_FIRE)
        # shape border
        shape = [(0, 0), (W-1, H-1)]
        draw.rectangle(shape, outline= COLOR_NON_FIRE, width=10)
        
        # calculate FPS
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = 'FPS: ' + str(fps)

        # show text
        draw.text((TEXTx, TEXTy + 34), fps, font = FONT, fill = TEXT_COLOR)
        draw.text((TEXTx, TEXTy), text, font = FONT, fill = TEXT_COLOR)

        frame = np.array(img_pil)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Tampilan', frame)
    cv2.waitKey(1)

    if cv2.getWindowProperty('Tampilan', cv2.WND_PROP_VISIBLE) < 1:
        break
    
    if DISPLAY_ONLY != True:
        if frame_counter % skip == 0:
            cv2.imwrite(IMAGE_PATH, frame)

        check_event(frame)

    if frame_counter == 100:
        frame_counter = 0
    else:
        frame_counter +=1
    
cap.release()
cv2.destroyAllWindows()


















