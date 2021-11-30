import cv2
import numpy as np
import pathlib

class Discriminator():
    def __init__(self):
        self.vid = None
        self.mode = None
        self.curr_frame = None
        self.curr_frame_original = None
        self.img_bin = None
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
        counter = 0
        for image in self.data_stamped:
            self.curr_frame = cv2.imread(str(image))

            self.curr_frame = cv2.resize(self.curr_frame, (0, 0), fx=0.15, fy=0.15)
            self.display_frame()
            self.curr_frame = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
            self.display_frame()
            _, self.img_bin = cv2.threshold(self.curr_frame, 100, 255, cv2.THRESH_OTSU)
            self.contours()

            #cv2.imshow('status', self.curr_frame)

            #self.display_frame()
            cv2.waitKey()

            cv2.imwrite(str(pathlib.Path(self.dir_stamped, str(counter) + '.jpg')), self.curr_frame)

            counter += 1

    def cropp_unstamped(self):
        cv2.imwrite(str(self.dir_unstamped), self.curr_frame)


    def contours(self):
        cnts, hierarchy = cv2.findContours(self.curr_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("binary", self.img_bin)
        for (cnt, hie) in zip(cnts, hierarchy[0]):
            print(hie)
            if hie[2] >= 0 and hie[3] == -1:  # contour has child(s) but no parent
                print('cnt found')

                if cv2.contourArea(cnt) > self.curr_frame.size / 4:  # kleine konturen ignorieren ( > 1/4 Bildfläche)
                    self.rect_min = cv2.minAreaRect(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    self.rect = x, y, w, h
                    box = cv2.boxPoints(self.rect_min)
                    self.box = np.int0(box)
                    # TODO# nice2hab trapezförmige bilder -> 4 eckpunkte finden

                    plot = cv2.rectangle(self.curr_frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 1)
                    # cv2.drawContours(plot,[box], 0, (255,0,0),1)
                    cv2.imshow('plot', plot)
                    #cv2.imwrite("demo.jpg", plot)

                    mask = self.img_bin.copy()  # np.zeros((h,w)) #! wert wird auch später immer auf 0 gesetzt
                    mask[:, :] = 0  # maske leeren
                    cv2.fillPoly(mask, [cnt], 255)
                    img_bin_roi = cv2.bitwise_and(self.img_bin[y:y + h, x:x + w],
                                                  mask[y:y + h, x:x + w])  # remove any other objects in roi
                    self.img_roi = self.img[y:y + h, x:x + w]  # für pixelzugriff mit schwerpunktkordinaten nötig
                    # cv2.imshow('mask', mask)
                    # cv2.imshow('bin_roi', img_bin_roi)
                    cv2.imshow('img roi', self.img_roi)
                    cv2.waitKey()



    def display_frame(self):
        cv2.imshow("current status of frame", self.curr_frame)

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




