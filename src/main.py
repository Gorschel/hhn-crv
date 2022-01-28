#!/usr/bin/env python
# -*- coding: utf-8 -*-

from discriminator import Discriminator
import pathlib

data_dir = pathlib.Path(pathlib.Path.cwd().parent, 'data')

data_stamped = list(data_dir.glob('stamped/*'))

data_unstamped = list(data_dir.glob('unstamped/*'))

pic_path = '../data/IMG_20211102_170950.jpg'
video_path = '../data/VID_20211102_171059.mp4'

disc_obj = Discriminator()

disc_obj.set_stamped_data(data_stamped)
disc_obj.set_unstamped_data((data_unstamped))
disc_obj.set_data_type(False)#False->stamped
disc_obj.crop_black_bg()
disc_obj.set_data_type(True)#True->unstamped
disc_obj.crop_black_bg()
