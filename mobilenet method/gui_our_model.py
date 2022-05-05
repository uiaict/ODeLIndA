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
from imageai.Prediction import imagenet_utils
from tensorflow.keras.models import load_model
import numpy as np
import time


class FunctionHolder:
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
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    available_ports.append(dev_port)
                else:
                    working_ports.append(dev_port)
            dev_port += 1
        return available_ports

    @staticmethod
    def getall_images():
        valid_file_endings = [".png", ".jpg"]
        onlyfiles = []
        all_images_tk = []
        for root, dirs, files in os.walk(os.path.abspath(os.getcwd() + "/obstruction/")):
            for file in files:
                onlyfiles.append(os.path.join(root, file))
        for i in range(len(onlyfiles)):
            img = Image.open(onlyfiles[i])
            resized_image = img.resize((150, 150), Image.ANTIALIAS)
            new_image = ImageTk.PhotoImage(resized_image)
            all_images_tk.append(new_image)
        return onlyfiles, all_images_tk

    @staticmethod
    def predictor_text(frame, labels, font, imwrite_text):
        # time.sleep(0.5)
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


def close(event):
    cv2.destroyAllWindows()
    sys.exit()


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


class App(tk.Tk):
    def __init__(self, title):
        super().__init__()
        self.title(title)
        self.geometry("1920x1080")


class CameraSelection_PopUp(Toplevel):
    def __init__(self, master, called_by, available_cams):
        Toplevel.__init__(self, master)
        self.master = master
        self.called = called_by
        self.geometry("300x300")
        self.called.stop = True
        self.master.withdraw()
        self.populate(available_cams)

    def populate(self, available_cams):
        # We only populate for a set number of cameras, max is 25 total camera options for a single program in thi case
        i = 0
        row = 0
        column = 1
        no_of_cam = len(available_cams)
        while i != no_of_cam:
            self.camera_button = Camera_Button(self, row, column - 1, available_cams[i], self)
            self.camera_button.configure(command=self.camera_button.set_var, text=available_cams[i])
            i += 1
            column += 1
            if column % 6 == 0:
                row += 1
                column = 0
        return 0

    def buttonpress(self):
        self.master.deiconify()
        self.destroy()


class Camera_Button(ttk.Button):
    def __init__(self, container, indexx, indexy, camera_no, parent):
        super().__init__(container)

        self.grid(column=indexx, row=indexy)
        # self.text = camera_no
        self.camera_no = camera_no
        self.parent = parent

    def set_var(self):
        self.parent.called.video_source_stream.set(self.camera_no)
        self.parent.buttonpress()


class ViewerFrame(ttk.Frame):
    def __init__(self, container, stop=True):
        super().__init__(container)
        options = {"padx": 5, "pady": 0}
        self.stop = stop
        self.counter = 0
        self.frameNumber = 0
        self.after_id = None
        """
        We store the camera source as an tk int
        """
        self.video_source_stream = tk.IntVar()
        self.video_source_stream.set(0)
        self.cap = cv2.VideoCapture(self.video_source_stream.get())
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.pause_unpause_var = tk.StringVar(value="Unpause")

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
        self.change_live_button.configure(command=lambda: self.callPopup(container))

        self.pause_unpause_button = ttk.Button(self, textvariable=self.pause_unpause_var)
        self.pause_unpause_button.grid(column=0, row=3, sticky="w")
        self.pause_unpause_button.configure(command=self.toggle_feed)
        # self.change_live_button.configure(command=self.toggle_feed)
        # Initiates the start of our frames being read
        # self.show_frames()
        # self.button_label = ttk.Label(self)
        # self.button_label.grid(column=0, row=1, sticky="w", **options)
        self.grid(column=0, row=0, padx=5, pady=5, sticky="nsew")

    def callPopup(self, master):
        self.stop = True
        self.cap.release()
        available_cams = FunctionHolder.get_Streams()

        if len(available_cams) == 0 or available_cams is None:
            showerror(title="Error", message="No available streams!")
        else:
            options_popup = CameraSelection_PopUp(master, self, available_cams)
            options_popup.wait_window()
        self.stop = False
        self.cap = cv2.VideoCapture(self.video_source_stream.get())
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def toggle_feed(self):
        if self.after_id == None:
            self.show_frames()
            self.pause_unpause_var.set("pause stream")
        else:
            self.after_cancel(self.after_id)
            self.after_id = None
            self.pause_unpause_var.set("unpause stream")

    def show_frames(self):

        """
        Method used to display the frames from our cap to the gui
        Currently best method as it does not stall the app during runtime
        However could be tidied up with the if statements
        """
        if not self.stop:
            start_time = time.time()
            fps = self.fps
            frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frameCount / fps
            target = fps / 5
            ret, frame = self.cap.read()
            height, width, channels = frame.shape
            frame_edited = cv2.flip(frame, 1)
            frame_edited = cv2.resize(frame_edited, (600, 600))
            cv2image = cv2.cvtColor(frame_edited, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            labels = ["Lens Crack  : Please repair camera lens",
                      "Dark view   : Drive carefully and decrease the speed",
                      "Dirty Lens  : Please clean the camera lens",
                      "Flared view : Please cover the view",
                      "Foggy View  : Drive carefully and decrease the speed",
                      "Frosted View: Clean the camera lens",
                      "Rainy View  : Drive carefully and decrease the speed",
                      ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.counter == target:
                # display and stuff
                self.counter = 0
                time_frame = str(timedelta(seconds=60 * (self.frameNumber / fps)))
                td = time_frame.split(':')
                if predictor(frame) == 1:
                    FunctionHolder.predictor_text(frame, labels[0], font, ["Lens Crack",
                                                                           f"Crack View at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}_crack.jpg"])
                elif predictor(frame) == 2:
                    FunctionHolder.predictor_text(frame, labels[1], font, ["Dark view",
                                                                           f"Dark View at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}_dark.jpg"])
                elif predictor(frame) == 3:
                    FunctionHolder.predictor_text(frame, labels[2], font, ["Dirty View",
                                                                           f"Dirty at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}_dirty.jpg"])
                elif predictor(frame) == 4:
                    FunctionHolder.predictor_text(frame, labels[3], font, ["Flare View",
                                                                           f"Flare at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}_flare.jpg"])
                elif predictor(frame) == 5:
                    FunctionHolder.predictor_text(frame, labels[4], font, ["Foggy View",
                                                                           f"Foggy at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}_foggy.jpg"])
                elif predictor(frame) == 6:
                    FunctionHolder.predictor_text(frame, labels[5], font, ["Frosted View",
                                                                           f"Frosted over at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}frosted.jpg"])
                elif predictor(frame) == 7:
                    FunctionHolder.predictor_text(frame, labels[6], font, ["Rainy View",
                                                                           f"Rainy at {td[1]}:{td[2]}",
                                                                           f"/obstruction/{td[1]}-{td[2]}_rainy.jpg"])
                else:
                    print(f"Clean View at {td[1]}:{td[2]}")
                self.frameNumber += 1
            else:
                ret = self.cap.grab()
                self.counter += 1
            self.after_id = self.video_label.after(5, self.show_frames)
            # print(time.time() - start_time)
        else:
            self.after_cancel(self.after_id)
            self.after_id = None

    """
    Method for changing the cap source to a video source
    The ChangeVideo.get_Video() method prompts the user to select a file
    If they give an invalid file we warn the user
    Otherwise it becomes the current cap content
    """

    def change_to_video(self, event=None):
        videopath = FunctionHolder.get_Video()
        if videopath is not None:
            self.cap = cv2.VideoCapture(videopath)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

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

    def show_image_button(self):
        img = cv2.imread(self.img_path)
        height, width, channel = img.shape
        if height > 1600 or width > 900:
            img = cv2.resize(img, (1600, 900))
        cv2.imshow("Image", img)


class AlbumFrame(ttk.Frame):
    def __init__(self, container, image_thresh, self_no=0, images=None, images_tk=None, is_child=False):
        super().__init__(container)
        self.grid(column=0, row=0, padx=5, sticky="nsew")
        self.parent = container
        self.images = images
        self.images_tk = images_tk
        # Threshold for nextpage on image album
        # only works for multiples of 5
        self.thresh = image_thresh
        self.is_child = is_child
        self.number = self_no
        self.pageno = tk.Label(self, text=str(self.number))
        self.pageno.grid(column=3, row=0)

    def populate(self):
        options = {"padx": 5, "pady": 0}
        image_increment = 0
        self.frames = {}
        if self.images is None:
            self.images, self.images_tk = FunctionHolder.getall_images()
        x_index = 1
        y_index = 0
        if self.is_child:
            self.change_live_button = ttk.Button(self, text="Prev page")
            self.change_live_button.grid(column=2, row=8, sticky="e")
            self.change_live_button.configure(command=lambda: self.change_frame(1))
            self.frames[1] = self.parent
        for each in self.images:
            icon = self.images_tk[image_increment]
            self.image_button = ButtonImage(self, x_index, y_index + 1, each, icon)
            self.image_button.configure(command=self.image_button.show_image_button, image=icon)
            # print("X index {}, Y index {}".format(x_index, y_index))
            if image_increment == self.thresh - 1:
                self.change_live_button = ttk.Button(self, text="Next page")
                self.change_live_button.grid(column=4, row=8, sticky="e")
                self.change_live_button.configure(command=lambda: self.change_frame(0))
                # print(len(self.images[self.thresh:]))
                self.frames[0] = AlbumFrame(self, image_thresh=25, is_child=True, images=self.images[self.thresh:],
                                            images_tk=self.images_tk[self.thresh:], self_no=self.number + 1)
                break
            x_index += 1
            image_increment += 1
            if x_index > 5:
                x_index = 1
                y_index += 1

    def change_frame(self, no):
        for child in self.winfo_children():
            # print("button" in str(child) or "")
            # print(str(child))
            if "button" in str(child):
                child.destroy()

        if no >= 0:
            frame = self.frames[no]
            # cheap way to suspend the first frame if it is being switched from
            frame.reset()
            frame.tkraise()

        #
        # for i in range(0, 5):
        #     for j in range(0, 5):
        #         icon = all_images_tk[image_increment]
        #         self.image_button = ButtonImage(self, i, j, all_images[image_increment], icon)
        #         self.image_button.configure(command=self.image_button.show_image_button, image=icon)
        #         # self.image_button.image = icon
        #         image_increment += 1

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
            text="Video feed",
            value=0,
            variable=self.selected_value,
            command=self.change_frame).grid(column=0, row=0, padx=5, pady=5)

        ttk.Radiobutton(
            self,
            text="Album",
            value=1,
            variable=self.selected_value,
            command=self.change_frame).grid(column=1, row=0, padx=5, pady=5)

        self.grid(column=0, row=1, padx=5, pady=5, sticky="ew")

        self.frames = {}
        self.frames[0] = ViewerFrame(
            container,
            False)
        self.frames[1] = AlbumFrame(
            container, image_thresh=25)
        self.change_frame()

    def change_frame(self):
        frame = self.frames[self.selected_value.get()]
        # cheap way to suspend the first frame if it is being switched from
        frame.reset()
        frame.tkraise()


if __name__ == "__main__":
    # load the model
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)
    if not os.path.isdir(cur_dir + r"\obstruction"):
        os.makedirs(cur_dir + r"\obstruction")
    else:
        print("Exists")

    if os.path.isdir(parent_dir + r"\mobilnet_model2.h5") or os.path.isdir(
            cur_dir + r"\mobilnet_model2.h5"):
        print("Missing model")
        exit()

    model = load_model('mobilnet_model2.h5')
    # summarize model.
    model.summary()
    app = App("Video")
    # esc to close the app
    app.bind('<Escape>', close)
    ControlFrame(app)
    app.mainloop()
