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

        self.img_flt = self.filter(self.img)
        self.img_ilm = self.fix_illum(self.img_flt)
        self.img_bin = self.threshhold(self.img_ilm)
        self.rect = None
        self.rect_min = None
        self.img_roi = None
        self.box = None
        self.cnts, self.hierarchy = self.contours(self.img_bin)
        self.img_warp = self.warpImage(self.img_bin)

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
            if hie[2] >= 0 and hie[3] == -1: # contour has child(s) but no parent
                if cv2.contourArea(cnt) > img.size/4: # kleine konturen ignorieren ( > 1/4 Bildfläche)
                    self.rect_min = cv2.minAreaRect(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    self.rect = x,y,w,h
                    box = cv2.boxPoints(self.rect_min)
                    self.box = np.int0(box)
                    #TODO# nice2hab trapezförmige bilder -> 4 eckpunkte finden

                    plot = cv2.rectangle(self.img.copy(),(x,y),(x+w,y+h),(0,255,0),1)
                    #cv2.drawContours(plot,[box], 0, (255,0,0),1)
                    cv2.imshow('plot', plot) 
                    cv2.imwrite("demo.jpg", plot)  

                    mask = self.img_bin.copy() #np.zeros((h,w)) #! wert wird auch später immer auf 0 gesetzt
                    mask[:,:] = 0 # maske leeren 
                    cv2.fillPoly(mask, [cnt], 255)
                    img_bin_roi = cv2.bitwise_and(self.img_bin[y:y+h, x:x+w], mask[y:y+h, x:x+w]) # remove any other objects in roi  
                    self.img_roi = self.img[y:y+h, x:x+w] # für pixelzugriff mit schwerpunktkordinaten nötig
                    #cv2.imshow('mask', mask)  
                    #cv2.imshow('bin_roi', img_bin_roi)
                    cv2.imshow('img roi', self.img_roi)
                    cv2.waitKey()
        
        return cnts, hierarchy

    def warpImage(self,img):
        x,y,w,h = self.box
        pts_dst = np.array([[x, y], [x, y+h], [x+w, y+h],[x+w, h]])
        h, _ = cv2.findHomography(self.box, pts_dst)
        return cv2.warpPerspective(self.img_roi, h,(self.img_roi.shape[1],self.img_roi.shape[0]))


if __name__ ==  '__main__':
    imgO = cv2.imread("data/IMG_20211102_171029.jpg")
    img = cv2.resize(imgO, (0, 0), fx=0.15, fy=0.15)
    Test = PreProcessing(img)
    cv2.waitKey()

