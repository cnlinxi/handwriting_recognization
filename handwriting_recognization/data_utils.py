# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 17:11
# @Author  : MengnanChen
# @FileName: data_utils.py
# @Software: PyCharm Community Edition

import os
import sys

sys.path.append(__file__)

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import np_utils
import tensorflow as tf
import scipy.misc
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import global_vals

# 压缩图片,把图片压缩成global_vals.resized_image_size*global_vals.resized_image_size的
def resize_img():
    resized_dir=os.path.join(global_vals.data_path,'resized')
    if not os.path.exists(resized_dir):
        os.mkdir(resized_dir)
    origin_data_dir=os.path.join(global_vals.data_path,'origin')
    if not os.path.exists(origin_data_dir):
        os.mkdir(origin_data_dir)
    filenames=os.listdir(origin_data_dir)
    for filename in filenames:
        im=tf.gfile.FastGFile(os.path.join(origin_data_dir,filename),'rb').read()
        with tf.Session() as sess:
            img_data=tf.image.decode_jpeg(im)
            image_float=tf.image.convert_image_dtype(img_data,tf.float32)
            resised=tf.image.resize_images(image_float,
                                           size=[global_vals.resized_image_size,global_vals.resized_image_size],method=3)
            resized_im=resised.eval()
            scipy.misc.imsave(os.path.join(resized_dir,filename),resized_im)

# 图片转h5文件
def image_to_h5():
    resized_dir=os.path.join(global_vals.data_path,'resized')
    if not os.path.exists(resized_dir):
        os.mkdir(resized_dir)
    resized_filenames = os.listdir(resized_dir)
    Y = []  # label
    X = []  # data
    for filename in resized_filenames:
        label = str(filename.split('_')[0]).encode()
        Y.append(label)
        im = Image.open(os.path.join(resized_dir,filename)).convert('RGB')
        mat = np.asarray(im)  # image 转矩阵
        X.append(mat)

    file = h5py.File(os.path.join(global_vals.data_path,'resized_image.h5'),'w')
    file.create_dataset('X', data=np.array(X))
    file.create_dataset('Y', data=np.array(Y))
    file.close()

# load dataset
def load_dataset():
    if not os.path.exists(global_vals.h5_file_path):
        print('generate h5 file...')
        resize_img()
        image_to_h5()
    data = h5py.File(global_vals.h5_file_path, 'r')
    X_data = np.array(data['X'])  # data['X']是h5py._hl.dataset.Dataset类型，转化为array
    Y_data = np.array(data['Y'])
    X_data = X_data / 255.
    # one-hot
    le=LabelEncoder()
    Y_data=le.fit_transform(Y_data)
    y_data = np_utils.to_categorical(Y_data, num_classes=global_vals.num_classes)

    return X_data, y_data

# 数据增强
# https://keras-cn.readthedocs.io/en/latest/preprocessing/image/
def data_augmentation():
    X_train, X_test=load_dataset()
    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator.fit(X_train)
    return train_generator

# h5 转 image
def h5_to_img():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()

    index = 1
    # 训练集
    train_dir = os.path.join(global_vals.data_path, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    for i in range(X_train_orig.shape[0]):
        # plt.imshow(X_train_orig[i])
        plt.imsave(fname=os.path.join(train_dir, '{}_{}.png'.format(Y_train_orig[0][i], str(index))),
                   arr=X_train_orig[i])
        index += 1

    # 测试集
    test_dir = os.path.join(global_vals.data_path, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i in range(X_test_orig.shape[0]):
        # plt.imshow(X_train_orig[i])
        plt.imsave(fname=os.path.join(test_dir, '{}_{}.png'.format(Y_test_orig[0][i], str(index))), arr=X_test_orig[i])
        index += 1


def split_file():
    dirs = os.listdir("generate r_pic")
    counter = 1
    index = 0
    for filename in dirs:
        if counter == 1:
            os.mkdir("split_pic//{}".format(str(index)))
        im = Image.open("generater_pic//{}".format(filename))
        im.save("split_pic//{}//{}".format(str(index), filename))
        counter += 1
        if counter == 2001:
            counter = 1
            index += 1

if __name__ == '__main__':
    resize_img()
    image_to_h5()