#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pathlib

class Discriminator():
    def __init__(self):
        self.vid = None
        self.data_type = None #Bool
        self.count = None
        self.mode = None
        self.curr_frame = None
        self.gray = None
        self.curr_frame_original = None
        self.img_bin = None
        self.stamped = None
        self.data_stamped = None
        self.data_unstamped = None
        self.dir_stamped = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data/stamped_cropped')
        self.dir_unstamped = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data/unstamped_cropped')
        self.dir_stamped_black = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data/stamped_cropped_black')
        self.dir_unstamped_black = pathlib.Path(pathlib.Path.home(), 'PycharmProjects/hhn-crv/data/unstamped_cropped_black')

    def set_mode(self, mode):
        self.mode = mode

    def set_data_type(self, mode):
        self.data_type = mode

    def set_stamped_data(self, stamped_data):
        self.data_stamped = stamped_data

    def set_unstamped_data(self, unstamped_data):
        self.data_unstamped = unstamped_data

    def crop_black_bg(self):
        self.count = 0
        if self.data_type:
            filelist = self.data_unstamped
        else:
            filelist = self.data_stamped
        print(len(filelist))

        for i, image in enumerate(filelist):
            self.count = i
            self.curr_frame = cv2.imread(str(image))

            self.curr_frame = cv2.resize(self.curr_frame, (0, 0), fx=0.5, fy=0.5)
            self.gray = self.curr_frame

            self.gray = cv2.cvtColor(self.gray, cv2.COLOR_BGR2GRAY)

            _, self.img_bin = cv2.threshold(self.gray, 100, 255, cv2.THRESH_OTSU)

            self.cut_black_bg(self.curr_frame)
            #cv2.imshow('status', self.curr_frame)

            #self.display_frame()
            #cv2.waitKey()

            #cv2.imwrite(str(pathlib.Path(self.dir_stamped, str(counter) + '.jpg')), self.curr_frame)



    def crop_minAreaRect(self, img, rect):
        sub_cnt = 0
        if self.data_type:
            dirname = self.dir_unstamped
        else:
            dirname = self.dir_stamped
        # rotate img
        angle = -rect[2]
        rows, cols = img.shape[0], img.shape[1]
        box_first = cv2.boxPoints(rect)
        box_first = np.int0(box_first)
        #cv2.drawContours(img, [box_first], 0, (255, 0, 0), 2)

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("rot", img_rot)

        grey_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(grey_rot, 100, 255, cv2.THRESH_OTSU)
        cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("bin", img_bin)

        for (cnt, hie) in zip(cnts, hierarchy[0]):

            if cv2.contourArea(cnt) > img.size / 30:  # kleine konturen ignorieren ( > 1/4 Bildfläche)
            #if hie[2] >= 0 and hie[3] < 0:  # contour has child(s) but no parent
                self.rect_min = cv2.boundingRect(cnt)
                #box = cv2.boxPoints(self.rect_min)
                #box = np.int0(self.rect_min)
                #cv2.drawContours(img_rot, [box], 0, (0, 255, 255), 2)

                x, y, w, h = self.rect_min
                #cv2.rectangle(img_rot, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("rahmen", img_rot)

                img_crop = img_rot[y:y+h, x:x+w]
                cv2.imshow("croped", img_crop)

                key = 0

                key = cv2.waitKey()
                self.count = self.count+1
                if key == 103: #g taste
                    #print(str(pathlib.Path(self.dir_stamped, str(self.count) + '.jpg')))
                    cv2.imwrite(str(pathlib.Path(dirname, str(self.count)+"_"+str(sub_cnt) + '.jpg')), img_crop)


                if key == 115: #s taste
                    print("bild schlecht")
            sub_cnt +=1
        # rotate bounding box
        #rect0 = (rect[0], rect[1], 0.0)
        #box = cv2.boxPoints(rect0)
        #draw_box = np.int0(box)
        #pts = np.int0(cv2.transform(np.array([box]), M))[0]
        #pts[pts < 0] = 0
        #print(draw_box)
        #cv2.drawContours(img_rot, [draw_box], 0, (0, 0, 255), 2)
        #cv2.drawContours(img_rot, [pts], 0, (0, 0, 255), 2)

        #cv2.imshow("croped", img_rot)

        # crop
        #img_crop = img_rot[draw_box[1][1]:draw_box[0][1], draw_box[1][0]:draw_box[2][0]]


        #self.curr_frame = img_crop

    def cropp_stamped(self):
        counter = 0
        for image in self.data_stamped:
            self.curr_frame = cv2.imread(str(image))

            self.curr_frame = cv2.resize(self.curr_frame, (0, 0), fx=0.5, fy=0.5)
            self.display_frame()
            grayimg = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
            #self.display_frame()
            _, self.img_bin = cv2.threshold(grayimg, 100, 255, cv2.THRESH_OTSU)
            #self.contours()
            self.cut_black_bg(self.img_bin)
            #cv2.imshow('status', self.curr_frame)

            #self.display_frame()
            cv2.waitKey()

            cv2.imwrite(str(pathlib.Path(self.dir_stamped, str(counter) + '.jpg')), self.curr_frame)

            counter += 1

    def cut_black_bg(self, img):
        #cv2.imshow("bin", self.img_bin)
        cnts, hierarchy = cv2.findContours(self.img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for (cnt, hie) in zip(cnts, hierarchy[0]):
            if cv2.contourArea(cnt) > img.size / 20:  # kleine konturen ignorieren ( > 1/4 Bildfläche)
                self.rect_min = cv2.minAreaRect(cnt)
                self.crop_minAreaRect(self.curr_frame, self.rect_min)

    def read_frame(self):
        ret, self.curr_frame = self.vid.read()
        self.curr_frame, self.curr_frame_original = cv2.resize(self.curr_frame, (0, 0), fx=0.15, fy=0.15)
        #cv2.imshow("read",self.curr_frame)

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




