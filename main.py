from tkinter import *
from tkvideo import tkvideo
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import timedelta
from PIL import Image
from PIL import ImageTk
import time

import cv2
import os
from tensorflow.keras.models import load_model
model = load_model('../best_model.h5')
# summarize model.
model.summary()
# load dataset
def predictor(img):
    test_image = cv2.resize(img, (200, 200))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image/255
    result=model.predict(test_image)
    if np.argmax(result) == 0:
        prediction = 0 #"There is no obstruction, it is clean view of camera"
    elif np.argmax(result) == 1:
        prediction = 1  #"There is a glass crack in view of camera"
    elif np.argmax(result) == 2:
        prediction = 2 #"There is a dirty view in the camera"
    elif np.argmax(result) == 3:
        prediction = 3 #"There is a foggy view in the camera"
    elif np.argmax(result) == 4:
        prediction = 4 #"There is a rainy view in the camera"
    print(np.argmax(result))
    return prediction

cap = cv2.VideoCapture("../test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount/fps
target = 5
counter = 0
frameNumber = 0
while True:
    if counter == target:
        ret, frame = cap.read()
        # display and stuff
        counter = 0
        if (predictor(frame) == 1):
            time_frame = str(timedelta(seconds=60*(frameNumber / fps)))
            td = time_frame.split(':')
            print(f"Crack View at {td[1]}:{td[2]}")
            cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_crack.jpg", frame)
        elif (predictor(frame) == 2):
            time_frame = str(timedelta(seconds=60*(frameNumber / fps)))
            td = time_frame.split(':')
            print(f"Dirty View at {td[1]}:{td[2]}")
            cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_dirty.jpg", frame)

        elif (predictor(frame) == 3):
            time_frame = str(timedelta(seconds=60*(frameNumber / fps)))
            td = time_frame.split(':')
            print(f"Foggy at {td[1]}:{td[2]}")
            cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_foggy.jpg", frame)
        elif (predictor(frame) == 4):
            time_frame = str(timedelta(seconds=60*(frameNumber / fps)))
            td = time_frame.split(':')
            print(f"Rainy at {td[1]}:{td[2]}")
            cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_rainy.jpg", frame)
        else:
            print(f"Clean View at {td[1]}:{td[2]}")
        frameNumber += 1
    else:
        ret = cap.grab()
        counter += 1


    #cv2.imshow("Video Frame",frame)
    #if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break



# When everything done, release the video capture object

cap.release()