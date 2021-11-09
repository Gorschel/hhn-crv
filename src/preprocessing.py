#!/usr/bin/env python

import cv2
import numpy as np

class PreProcessing(object):
    """
    contains all preprocessing methods

    """
    def __init__(self, img):

        self.img = img

        #self.claheimg = self.clahe()
        #cv2.imshow("clahe", self.claheimg)

        self.flt = self.filter(self.img)
        self.ilm = self.fix_illum(self.flt)

        self.binimg = self.threshhold(self.ilm)
        cv2.imshow("bin", self.binimg)
        self.cnts, self.hierarchy = self.contours(self.binimg)
        roi = self.find_roi(self.binimg)

    def clahe(self):
        # BGR 2 Lab
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        # split L channel and equalize hist
        L, a, b = cv2.split(lab)
        Ln = cv2.equalizeHist(L)
        # combine and convert 2 BGR
        return cv2.cvtColor(cv2.merge((Ln, a, b)), cv2.COLOR_LAB2BGR)

    def filter(self,img):
        grimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grimg

    def fix_illum(self,img):
        return img

    def threshhold(self,img):
        _,binimg = cv2.threshold(img,100,255,cv2.THRESH_OTSU)
        return binimg

    def contours(self,img):
        cnts, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        
        for (cnt, hie) in zip(cnts, hierarchy[0]):      
            if hie[2] >= 0 and hie[3] < 0: # contour has child(s) but no parent
                x,y,w,h = cv2.boundingRect(cnt)
                plot = cv2.rectangle(self.binimg,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('plot', plot)
                               
                mask = self.binimg.copy() #np.zeros((h,w)) #! wert wird auch später immer auf 0 gesetzt
                mask[:,:] = 0 # maske leeren 
                cv2.fillPoly(mask, [cnt], 255)
                img_bin_roi = cv2.bitwise_and(self.binimg[y:y+h, x:x+w], mask[y:y+h, x:x+w]) # remove any other objects in roi  
                img_roi = self.img[y:y+h, x:x+w] # für pixelzugriff mit schwerpunktkordinaten nötig
                #cv2.imshow('mask', mask)  
                cv2.imshow('bin_roi', img_bin_roi)
                cv2.imshow('img roi', img_roi)
        
        return cnts, hierarchy

    def find_roi(self,img):
        return img


if __name__ ==  '__main__':
    imgO = cv2.imread("data/IMG_20211102_171029.jpg")
    img = cv2.resize(imgO, (0, 0), fx=0.15, fy=0.15)
    Test = PreProcessing(img)
    cv2.waitKey()

