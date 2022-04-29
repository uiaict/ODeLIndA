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
import glob

model = load_model(r"mobilnet_model2.h5")

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
          "Rainy View  : Drive carefully and decrease the speed",]

font = cv2.FONT_HERSHEY_SIMPLEX

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
    
which_mode = int(input("1 -> live, 2-> dir"))

if (which_mode == 1):
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        if predictor(frame) == 1:
            print("lens crack")
    
        elif predictor(frame) == 2:
            print("dark view")
        elif predictor(frame) == 3:
            print("dirty view")
        elif predictor(frame) == 4:
            print("flare view")
        elif predictor(frame) == 6:
            print("frosted view")
        elif predictor(frame) == 7:
            print("rainy view")
        else:
            print(f"Clean View at " + os.path.basename(each))
    
else:
    base_path = r"E:\programming\bachelor_project\BachelorOppgave\mobilenet method\test_images\\"

    basefiles = []
    integer = 0
    for files  in glob.glob(base_path + r"*"):
        basefiles.append(files)

    for each in basefiles:
        frame = cv2.imread(each)
        
        if predictor(frame) == 1:
            predictor_text(frame, labels[0], font, ["Lens Crack",
                                                    f"Crack View",
                                                    f"/obstruction/{td[1]}-{td[2]}_crack.jpg"])
        elif predictor(frame) == 2:
            predictor_text(frame, labels[1], font, ["Dark view",
                                                    f"Dark View",
                                                    f"/obstruction/" + os.path.basename(each) + "_dark.jpg"])
        elif predictor(frame) == 3:
            predictor_text(frame, labels[2], font, ["Dirty View",
                                                    f"Dirty at" + os.path.basename(each),
                                                    f"/obstruction/" + os.path.basename(each) + "_dirty.jpg"])
        elif predictor(frame) == 4:
            predictor_text(frame, labels[3], font, ["Flare View",
                                                    f"Flare at " + os.path.basename(each),
                                                    f"/obstruction/" + os.path.basename(each) + "_flare.jpg"])
        elif predictor(frame) == 6:
            FunctionHolder.predictor_text(frame, labels[5], font, ["Frosted View",
                                                                   f"Frosted over at" + os.path.basename(each) ,
                                                                   f"/obstruction/" + os.path.basename(each) + "_frosted.jpg"])
        elif predictor(frame) == 7:
            FunctionHolder.predictor_text(frame, labels[6], font, ["Rainy View",
                                                                   f"Rainy at {td[1]}:{td[2]}",
                                                                   f"/obstruction/" + os.path.basename(each) + "_rainy.jpg"])
        else:
            print(f"Clean View at " + os.path.basename(each))
    
