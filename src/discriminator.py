import cv2 as cv
import cv2.cv2
import numpy as np
import pathlib

class Discriminator():
    def __init__(self):
        self.vid = None
        self.mode = None
        self.curr_frame = None
        self.stamped = None
        self.data_stamped = None
        self.data_unstamped = None
        self.dir_stamped = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data/stamped_cropped')
        self.dir_unstamped = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data/unstamped_cropped')

    def set_mode(self, mode):
        self.mode = mode

    def set_stamped_data(self, stamped_data):
        self.data_stamped = stamped_data

    def set_unstamped_data(self, unstamped_data):
        self.data_unstamped = unstamped_data

    def cropp_stamped(self):
        for image in self.data_stamped:
            self.curr_frame = cv2.imread(str(image))
            cv2.imwrite(str(self.dir_stamped), self.curr_frame)

    def cropp_unstamped(self):
        cv2.imwrite(str(self.dir_unstamped), self.curr_frame)


    def display_frame(self):
        cv.imshow("current status of frame", self.curr_frame)

    def read_frame(self):
        ret, self.curr_frame = self.vid.read()
        self.curr_frame = cv.resize(self.curr_frame, (0, 0), fx=0.15, fy=0.15)
        #cv.imshow("read",self.curr_frame)

    def set_vid(self, vid):
        self.vid = vid

        cv2.imwrite('',self.curr_frame)


    def release_vid(self):
        self.vid.release()

    def illumination(self, alpha, beta):

        #for y in range(self.curr_frame.shape[0]):
        #    print(y)
        #    for x in range(self.curr_frame.shape[1]):
        #        for c in range(self.curr_frame.shape[2]):
        #            self.curr_frame[y, x, c] = np.clip(alpha * self.curr_frame[y, x, c] + beta[c], 0, 255)

        self.display_frame()




