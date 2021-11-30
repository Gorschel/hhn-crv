import cv2
import numpy as np

class test (object):
    def __init__(self):
        self.data = None
        self.mode = None
        self.img = None

    def set_mode(self, mode:int, path:str):
        """
        modes:
        0: single picture
        1: video file
        2: live
        """
        last = self.mode
        self.mode = mode
        if mode == 0:
            pass
        if mode == 1:
            print("videomode")
            self.data = cv2.VideoCapture(path)
        if mode == 2:
            pass

    def display_frame(self, name="current status of frame"):
        cv2.imshow(name, self.img)

    def read_frame(self):
        ret, self.curr_frame = self.vid.read()
        self.curr_frame = cv2.resize(self.curr_frame, (0, 0), fx=0.15, fy=0.15)
        #cv.imshow("read",self.curr_frame)

    def set_vid(self, vid):
        self.vid = vid

    def release_vid(self):
        self.vid.release()

    def illumination(self, alpha, beta):

        #for y in range(self.curr_frame.shape[0]):
        #    print(y)
        #    for x in range(self.curr_frame.shape[1]):
        #        for c in range(self.curr_frame.shape[2]):
        #            self.curr_frame[y, x, c] = np.clip(alpha * self.curr_frame[y, x, c] + beta[c], 0, 255)

        self.display_frame()

