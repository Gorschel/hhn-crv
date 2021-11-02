import cv2 as cv
import numpy as np

from discriminator import Discriminator

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
    disc_obj.curr_frame = cv.resize(cv.imread(pic_path, 1), (0, 0), fx=0.15, fy=0.15)
    disc_obj.illumination(alpha=c0, beta=beta)
    cv.waitKey()

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

else:
    disc_obj.set_vid(cv.VideoCapture(0))
    while True:
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        disc_obj.illumination()

    disc_obj.release_vid()





