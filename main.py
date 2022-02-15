import tkinter as tk
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askopenfilename
import sys
from tkinter import Toplevel
from tkinter.messagebox import showerror
from PIL import Image, ImageTk
from tkinter import ttk
import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np


def predictor(img):
    test_image = cv2.resize(img, (200, 200))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image/255
    result=model.predict(test_image)
    prediction = 0
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



class ChangeVideo:
    @staticmethod
    def get_Video():
        # We can expand valid file endings but this is all of them for now
        # Otherwise this prompts the user to get a file to play in the GUI
        valid_file_endings = [".mp4"]
        result = askopenfilename()
        if result[-4:] in valid_file_endings:
            return result
        else:
            showerror(title="Error", message="Invalid file, currently we support %s" % "".join(valid_file_endings))

    @staticmethod
    # Method that tries to find a valid USB device to read images from
    # until it has more than 5 failures
    def get_Streams():
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []
        while len(non_working_ports) < 6:
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
                # print("Port %s is not working." % dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    # print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                    available_ports.append(dev_port)
                else:
                    # print("Port %s for camera (%s x %s) is present but does not reads." % (dev_port, h, w))
                    working_ports.append(dev_port)
            dev_port += 1
        return available_ports, working_ports, non_working_ports

    @staticmethod
    def getall_images():
        valid_file_endings = [".png", ".jpg"]
        onlyfiles = []
        all_images_tk = []
        for root, dirs, files in os.walk(os.path.abspath("../obstruction/")):
            for file in files:
                onlyfiles.append(os.path.join(root, file))
        for i in range(len(onlyfiles)):
            img = Image.open(onlyfiles[i])
            resized_image = img.resize((150, 150), Image.ANTIALIAS)
            new_image = ImageTk.PhotoImage(resized_image)
            all_images_tk.append(new_image)

        return onlyfiles, all_images_tk


def close(event):
    sys.exit()


# def onClick():
#     inputDialog = MyDialog(app)
#     app.wait_window(inputDialog.top)
#     print(result)

class App(tk.Tk):
    def __init__(self, title):
        super().__init__()
        self.title(title)
        self.geometry("1920x1080")


class MyDialog:
    def __init__(self, parent):
        top = self.top = Toplevel(parent)
        self.myLabel = tk.Label(top, text="Enter your username below")
        self.myLabel.pack()
        self.send()

    def send(self):
        global result
        result = None
        self.top.destroy()


class ViewerFrame(ttk.Frame):
    def __init__(self, container, stop=True):
        super().__init__(container)
        options = {"padx": 5, "pady": 0}
        self.stop = stop
        self.counter = 0
        self.frameNumber = 0
        """
        We store the camera source and video file source differently
        as tk variables are strict
        """
        self.video_source_file = tk.StringVar()
        self.video_source_stream = tk.IntVar()
        self.video_source_stream.set(0)
        if not self.stop:
            self.cap = cv2.VideoCapture(self.video_source_stream.get())
            self.width = 800
            self.height = 600
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Label holder for our video images
        self.video_label = ttk.Label(self)
        self.video_label.grid(column=0, row=0, sticky="w", **options)

        # Button to change source to a video
        self.change_source_button = ttk.Button(self, text="Change to video source")
        self.change_source_button.grid(column=0, row=1, sticky="w")
        self.change_source_button.configure(command=self.change_to_video)
        # Button to change source to a camera (currently only works on source = 0)
        self.change_live_button = ttk.Button(self, text="Change to live source")
        self.change_live_button.grid(column=0, row=2, sticky="w")
        self.change_live_button.configure(command=self.change_to_live)
        # Initiates the start of our frames being read
        self.show_frames()
        # self.button_label = ttk.Label(self)
        # self.button_label.grid(column=0, row=1, sticky="w", **options)
        self.grid(column=0, row=0, padx=5, pady=5, sticky="nsew")

    def show_frames(self):
        """
        Method used to display the frames from our cap to the gui
        Currently best method as it does not stall the app during runtime
        However could be tidied up with the if statements
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frameCount / fps
        target = 5
        if not self.stop:
            ret, frame = self.cap.read()
            height, width, channels = frame.shape
            frame_edited = cv2.flip(frame, 1)
            if (height >= 1920 or width >= 1080):
                frame_edited = cv2.resize(frame, (900, 900))
            cv2image = cv2.cvtColor(frame_edited, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            if self.counter == target:
                # display and stuff
                self.counter = 0
                time_frame = str(timedelta(seconds=60 * (self.frameNumber / fps)))
                td = time_frame.split(':')
                if (predictor(frame) == 1):
                    print(f"Crack View at {td[1]}:{td[2]}")
                    cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_crack.jpg", frame)
                elif (predictor(frame) == 2):
                    print(f"Dirty View at {td[1]}:{td[2]}")
                    cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_dirty.jpg", frame)
                elif (predictor(frame) == 3):
                    print(f"Foggy at {td[1]}:{td[2]}")
                    cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_foggy.jpg", frame)
                elif (predictor(frame) == 4):
                    print(f"Rainy at {td[1]}:{td[2]}")
                    cv2.imwrite(f"../obstruction/{td[1]}-{td[2]}_rainy.jpg", frame)
                else:
                    print(f"Clean View at {td[1]}:{td[2]}")
                self.frameNumber += 1
            else:
                ret = self.cap.grab()
                self.counter += 1
            self.video_label.after(4, self.show_frames)

    """
    Method for changing the cap source to a video source
    The ChangeVideo.get_Video() method prompts the user to select a file
    If they give an invalid file we warn the user
    Otherwise it becomes the current cap content
    """
    def change_to_video(self, event=None):
        videopath = ChangeVideo.get_Video()
        if videopath is not None:
            self.cap = cv2.VideoCapture(videopath)
        self.stop = False

    """
    Method being worked on, will prompt the user with a custom dialog
    The dialog will be buttons for the valid video sources
    In case they have multiple sources
    That is what is planned but for now it only switches to the 0th video source
    """
    def change_to_live(self, event=None):
        # Planned code for the future
        # Will implement as we move on but for now wanted a working solution
        # Rather than one that accounts for all scenarios
        # self.stop = True
        # self.cap.release()
        # available_ports, working_ports, non_working_ports = ChangeVideo.get_Streams()
        # print(available_ports)
        # onClick()
        self.cap = cv2.VideoCapture(self.video_source_stream.get())
        self.stop = False

    # Resets the frame on being raised to play the video again
    def reset(self):
        self.counter = 0
        self.frameNumber = 0
        self.stop = False



class ButtonImage(ttk.Button):
    def __init__(self, container, indexx, indexy, image_path, pil_image):
        super().__init__(container)

        self.grid(column=indexx, row=indexy)

        self.img_path = image_path
        self.image = pil_image

    def print_text(self):
        img = cv2.imread(self.img_path)
        height, width, channel = img.shape
        if height > 1600 or width > 900:
            img = cv2.resize(img, (1600, 900))
        cv2.imshow("Image", img)


class AlbumFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        self.grid(column=0, row=0, padx=5, sticky="nsew")

    def populate(self):
        options = {"padx": 5, "pady": 0}
        image_increment = 0
        all_images, all_images_tk = ChangeVideo.getall_images()
        for i in range(0, 5):
            for j in range(0, 5):
                icon = all_images_tk[image_increment]
                self.image_button = ButtonImage(self, i, j, all_images[image_increment], icon)
                self.image_button.configure(command=self.image_button.print_text, image=icon)
                #self.image_button.image = icon
                image_increment += 1

        # self.change_source_button = ttk.Button(self, text="View image")
        # self.change_source_button.grid(column=0, row=0)
        # self.change_source_button.configure(command=lambda: print("a"))

    def reset(self):
        for child in self.winfo_children():
            child.destroy()
        self.populate()


class ControlFrame(ttk.LabelFrame):
    def __init__(self, container):
        super().__init__(container)
        self["text"] = "Options"

        self.selected_value = tk.IntVar()

        ttk.Radiobutton(
            self,
            text="Display Video",
            value=0,
            variable=self.selected_value,
            command=self.change_frame).grid(column=0, row=0, padx=5, pady=5)

        ttk.Radiobutton(
            self,
            text="Display Video",
            value=1,
            variable=self.selected_value,
            command=self.change_frame).grid(column=1, row=0, padx=5, pady=5)

        self.grid(column=0, row=1, padx=5, pady=5, sticky="ew")

        self.frames = {}
        self.frames[0] = ViewerFrame(
            container,
            False)
        self.frames[1] = AlbumFrame(
            container)

        self.change_frame()

    def change_frame(self):
        frame = self.frames[self.selected_value.get()]
        # cheap way to suspend the first frame if it is being switched from
        frame.reset()
        frame.tkraise()


if __name__ == "__main__":
    # load the model
    model = load_model('best_model.h5')
    # summarize model.
    model.summary()
    app = App("Video")
    # esc to close the app
    app.bind('<Escape>', close)
    ControlFrame(app)
    app.mainloop()
