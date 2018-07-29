# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 17:16
# @Author  : MengnanChen
# @FileName: global_vals.py
# @Software: PyCharm Community Edition

import os

# data
variant_per_picture=200
data_path='data'
origin_picture_root_dir='data'
h5_file_path=os.path.join(origin_picture_root_dir,'resized_image.h5')
resized_image_size=64
output_dim_vectors=200 # the dimension of final output feature vector
num_classes=15 # how many people in the training set in handwriting recognition
