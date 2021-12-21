#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

from discriminator import Discriminator
import pathlib

data_dir = pathlib.Path(pathlib.Path.cwd(), 'data')

data_stamped = list(data_dir.glob('stamped/*'))

data_unstamped = list(data_dir.glob('unstamped/*'))

pic_path = '../data/IMG_20211102_170950.jpg'
video_path = '../data/VID_20211102_171059.mp4'

c0 = 1
cr = 0
cg = 0
cb = 0
beta = [cr, cg, cb]

disc_obj = Discriminator()

disc_obj.set_mode(mode=0)
# mode=0, pic
#     =1, video
#     =2, live

if disc_obj.mode == 0:
    print("picmode")
    #disc_obj.curr_frame = cv.resize(cv.imread(pic_path, 1), (0, 0), fx=0.5, fy=0.5)
    #disc_obj.illumination(alpha=c0, beta=beta)
    #cv.waitKey()

elif disc_obj.mode == 1:
    print("videomode")
    disc_obj.set_vid(cv.VideoCapture(video_path))
    while True:
        disc_obj.read_frame()
        disc_obj.illumination()
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
    disc_obj.release_vid()

elif disc_obj.mode == 2:
    pass

else:
    disc_obj.set_vid(cv.VideoCapture(0))
    while True:
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        disc_obj.illumination()

    disc_obj.release_vid()

disc_obj.set_stamped_data(data_stamped)
disc_obj.set_unstamped_data((data_unstamped))
disc_obj.set_data_type(False)#False->stamped
disc_obj.crop_black_bg()
disc_obj.set_data_type(True)#True->unstamped
disc_obj.crop_black_bg()
#disc_obj.cropp_unstamped()
