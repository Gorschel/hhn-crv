import cv2 as cv
import numpy as np

class Discriminator():
    def __init__(self):
        self.vid = None
        self.mode = None
        self.curr_frame = None

    def set_mode(self, mode):
        self.mode = mode

    def display_frame(self):
        cv.imshow("current status of frame", self.curr_frame)

    def read_frame(self):
        ret, self.curr_frame = self.vid.read()
        self.curr_frame = cv.resize(self.curr_frame, (0, 0), fx=0.15, fy=0.15)
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




