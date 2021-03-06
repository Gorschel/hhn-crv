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
        self.curr_frame = None
        self.gray = None
        self.curr_frame_original = None
        self.img_bin = None
        self.img_bin1 = None
        self.img_binmorph1 = None
        self.img_rot_col = None
        self.img_grey_bg = None
        self.img_bin2 = None
        self.img_binmorph2 = None
        self.stamped = None
        self.data_stamped = None
        self.data_unstamped = None
        self.dir_stamped = pathlib.Path(pathlib.Path.cwd().parent, 'data/stamped_cropped')
        self.dir_unstamped = pathlib.Path(pathlib.Path.cwd().parent, 'data/unstamped_cropped')
        self.dir_errrors = pathlib.Path(pathlib.Path.cwd().parent, 'data/cropping_errors')
        self.dir_stamped_black = pathlib.Path(pathlib.Path.cwd().parent, 'data/stamped_cropped_black')
        self.dir_unstamped_black = pathlib.Path(pathlib.Path.cwd().parent, 'data/unstamped_cropped_black')

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
            self.img_bin1 = self.img_bin
            self.cut_black_bg(self.curr_frame)

    def crop_minAreaRect(self, img, rect):
        sub_cnt = 0
        cv2.imshow("start img", img)
        if self.data_type:
            dirname = self.dir_unstamped
        else:
            dirname = self.dir_stamped

        angle = -rect[2]
        rows, cols = img.shape[0], img.shape[1]
        box_first = cv2.boxPoints(rect)
        box_first = np.int0(box_first)
        M_dreh = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        img_rot = cv2.warpAffine(img, M_dreh, (cols, rows))
        self.img_rot_col = img_rot
        grey_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

        height, width, _ = img_rot.shape
        lum_mean = cv2.mean(grey_rot)
        for i in range(height):
            for j in range(width):
                if grey_rot[i, j] < 5:
                    grey_rot[i, j] = lum_mean[0]

        self.img_grey_bg = grey_rot
        _, img_bin = cv2.threshold(grey_rot, 150, 255, cv2.THRESH_OTSU)
        self.img_bin2 = img_bin
        morph_img = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1, borderType=cv2.MORPH_RECT)
        self.img_binmorph2 = morph_img
        cnts, hierarchy = cv2.findContours(morph_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("bin2_morph", morph_img)

        print("bild #: "+str(self.count)+", anz konturen: "+str(len(cnts)))
        pic_good = False
        for (cnt, hie) in zip(cnts, hierarchy[0]):
            if cv2.contourArea(cnt) > img.size / 100:  # kleine konturen ignorieren ( > 1/4 Bildfl??che)
                if hie[3] < 0:  # contour has no parent
                    self.rect_min = cv2.boundingRect(cnt)
                    x, y, w, h = self.rect_min
                    img_crop = img_rot[y:y+h, x:x+w]
                    cv2.imshow("croped", img_crop)
                    key = 0
                    key = cv2.waitKey()
                    self.count = self.count+1
                    pic_good = True
                    if key == 103: #g taste
                        cv2.imwrite(str(pathlib.Path(dirname, str(self.count)+"_"+str(sub_cnt) + '.jpg')), img_crop)
                        print("bild gut")
                        break
                    if key == 98:  # b taste
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count)+"_"+str(sub_cnt) + 'bin1'+'.jpg')) , self.img_bin1)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'start' + '.jpg')),img)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'bin1morph' + '.jpg')),self.img_binmorph1)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'bin2' + '.jpg')),self.img_bin2)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'bin2morph' + '.jpg')),self.img_binmorph2)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'rot_col'+'.jpg')), self.img_rot_col)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'grey_bg'+'.jpg')), self.img_grey_bg)
                        cv2.imwrite(str(pathlib.Path(self.dir_errrors, str(self.count) + "_" + str(sub_cnt) + 'cropped' + '.jpg')),img_crop)
                        print("error saved")
                else:
                    print("hat parent: "+str(sub_cnt))
                sub_cnt += 1
            else:
                print("zu klein: "+ str(sub_cnt))
                sub_cnt += 1
        if not pic_good:
            print(str(self.count)+" liefert nix")

    def cropp_stamped(self):
        counter = 0
        for image in self.data_stamped:
            self.curr_frame = cv2.imread(str(image))
            self.curr_frame = cv2.resize(self.curr_frame, (0, 0), fx=0.5, fy=0.5)
            self.display_frame()
            grayimg = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
            _, self.img_bin = cv2.threshold(grayimg, 100, 255, cv2.THRESH_OTSU)
            self.cut_black_bg(self.img_bin)
            cv2.waitKey()
            cv2.imwrite(str(pathlib.Path(self.dir_stamped, str(counter) + '.jpg')), self.curr_frame)
            counter += 1

    def cut_black_bg(self, img):
        morph_img = cv2.morphologyEx(self.img_bin, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1, borderType=cv2.MORPH_RECT)
        self.img_binmorph1 = morph_img
        cnts, hierarchy = cv2.findContours(morph_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        max_cnt = cnts[max_index]
        if cv2.contourArea(max_cnt) > img.size / 100:  # kleine konturen ignorieren ( > 1/4 Bildfl??che)
            self.rect_min = cv2.minAreaRect(max_cnt)
            self.crop_minAreaRect(self.curr_frame, self.rect_min)
        else:
            print("!!!!!!!!!!!!!!!!!!!!zu klein, kann nicht rotieren")
            pass

    def read_frame(self):
        ret, self.curr_frame = self.vid.read()
        self.curr_frame, self.curr_frame_original = cv2.resize(self.curr_frame, (0, 0), fx=0.15, fy=0.15)
        self.curr_frame_original = self.curr_frame

    def set_vid(self, vid):
        self.vid = vid
        cv2.imwrite('', self.curr_frame)

    def release_vid(self):
        self.vid.release()