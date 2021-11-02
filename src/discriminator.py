import cv2 as cv
import argparse
import numpy as np

class Discriminator():
    def __init__(self):

        self.vid = None
        self.mode = None

        self.curr_frame = None

    def set_mode(self, mode):
        self.mode = mode

    def read_frame(self):
        ret, self.curr_frame = self.vid.read()
        #cv.imshow("read",self.curr_frame)

    def set_vid(self, vid):
        self.vid = vid

    def release_vid(self):
        self.vid.release()




