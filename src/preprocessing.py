#!/usr/bin/env python

import cv2
import numpy as np

class PreProcessing(object):
    """
    contains all preprocessing methods

    """
    def __init__(self, img):

        self.img = img
        self.clahe = self.clahe()
        cv2.imshow("clahe", self.clahe)


        """
        self.flt = self.filter(img)
        self.ilm = self.fix_illum(self.flt)


        self.bin = self.threshhold(self.ilm)
        self.cnt = self.contours(self.bin)
        """

    def clahe(self):
        # BGR 2 Lab
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        # split L channel and equalize hist
        L, a, b = cv2.split(lab)
        Ln = cv2.equalizeHist(L)
        # combine and convert 2 BGR
        return cv2.cvtColor(cv2.merge((Ln, a, b)), cv2.COLOR_LAB2BGR)

    def filter(self,img):
        return img

    def fix_illum(self,img):
        return img

    def threshhold(self,img):
        return img

    def contours(self,img):
        return img

    def find_roi(self,img):
        return img


if __name__ ==  '__main__':
    img = cv2.imread("../data/IMG_20211102_171029.jpg")
    Test = PreProcessing(img)

