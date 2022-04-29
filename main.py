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

model = load_model(r"mobilenet method\mobilnet_model2.h5")
# summarize model.
model.summary()


# load dataset
def predictor(img):
    test_image = cv2.resize(img, (224, 224))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255
    result = model.predict(test_image)
    prediction = 0
    if np.argmax(result) == 0:
        prediction = 0  # "There is no obstruction, it is clean view of camera"
    elif np.argmax(result) == 1:
        prediction = 1  # "There is a lens crack in view of camera"
    elif np.argmax(result) == 2:
        prediction = 2  # "There is a dark view in the camera"
    elif np.argmax(result) == 3:
        prediction = 3  # "There is a dirty view in the camera"
    elif np.argmax(result) == 4:
        prediction = 4  # "There is a flare view in the camera"
    elif np.argmax(result) == 5:
        prediction = 5  # "There is a foggy view in the camera"
    elif np.argmax(result) == 6:
        prediction = 6  # "There is a frost view in the camera"
    elif np.argmax(result) == 7:
        prediction = 7  # "There is a rainy view in the camera"
    # print(np.argmax(result))
    return prediction


def predictor_text(frame, labels, font, imwrite_text):
    # time.sleep(0.5)
    print(labels)
    cv2.putText(frame,
                labels,
                (600, 50), font
                , 1,
                (0, 0, 255),
                2,
                cv2.LINE_AA)
    percent = 70
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    frame25 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(imwrite_text[0], frame25)
    cv2.imwrite(os.curdir + imwrite_text[2], frame)
    print(imwrite_text[1])
    # print(imwrite_text[2])


labels = ["Lens Crack  : Please repair camera lens",
          "Dark view   : Drive carefully and decrease the speed",
          "Dirty Lens  : Please clean the camera lens",
          "Flared view : Please cover the view",
          "Foggy View  : Drive carefully and decrease the speed",
          "Frosted View: Clean the camera lens",
          "Rainy View  : Drive carefully and decrease the speed",
          ]

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount / fps
target = 5
counter = 0
frameNumber = 0
while True:
    cur_time = time.time()
    if counter == target:
        ret, frame = cap.read()
        # display and stuff
        counter = 0
        time_frame = str(timedelta(seconds=60 * (frameNumber / fps)))
        td = time_frame.split(':')
        if predictor(frame) == 1:
            predictor_text(frame, labels[0], font, ["Lens Crack",
                                                    f"Crack View at {td[1]}:{td[2]}",
                                                    f"/obstruction/{td[1]}-{td[2]}_crack.jpg"])
        elif predictor(frame) == 2:
            predictor_text(frame, labels[1], font, ["Dark view",
                                                    f"Dark View at {td[1]}:{td[2]}",
                                                    f"/obstruction/{td[1]}-{td[2]}_dark.jpg"])
        elif predictor(frame) == 3:
            predictor_text(frame, labels[2], font, ["Dirty View",
                                                    f"Dirty at {td[1]}:{td[2]}",
                                                    f"/obstruction/{td[1]}-{td[2]}_dirty.jpg"])
        elif predictor(frame) == 4:
            predictor_text(frame, labels[3], font, ["Flare View",
                                                    f"Flare at {td[1]}:{td[2]}",
                                                    f"/obstruction/{td[1]}-{td[2]}_flare.jpg"])
        elif predictor(frame) == 5:
            predictor_text(frame, labels[4], font, ["Foggy View",
                                                    f"Foggy at {td[1]}:{td[2]}",
                                                    f"/obstruction/{td[1]}-{td[2]}_foggy.jpg"])
        elif predictor(frame) == 6:
            predictor_text(frame, labels[5], font, ["Rainy View",
                                                    f"Rainy at {td[1]}:{td[2]}",
                                                    f"/obstruction/{td[1]}-{td[2]}_rainy.jpg"])
        else:
            print(f"Clean View at {td[1]}:{td[2]}")
        frameNumber += 1
    else:
        ret = cap.grab()
        counter += 1
    print(time.time() - cur_time)

    # cv2.imshow("Video Frame",frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break

# When everything done, release the video capture object

cap.release()


