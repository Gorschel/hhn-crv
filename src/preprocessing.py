#!/usr/bin/env python

import cv2
import numpy as np

class Image(object):
    def __init__(self, img, name):
        self.img = img
        self.name = str(name)

    def show(self):
        cv2.imshow(self.name , self.img)

class PreProcessing(object):
    """
    contains all preprocessing methods

    """
    def __init__(self, img):
        self.img = img
        self.flt = self.filter(img)
        self.ilm = self.fix_illum(self.flt)
        self.bin = self.threshhold(self.ilm)
        self.cnt = self.contours(self.bin)

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
    pass
