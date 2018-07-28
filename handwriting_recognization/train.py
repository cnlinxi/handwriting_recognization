# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 19:51
# @Author  : MengnanChen
# @FileName: train.py
# @Software: PyCharm Community Edition

import os
import sys
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
import time
from tensorflow.python.framework import graph_util

import global_vals
import data_utils
from capsule_network import capsule


def weight_variable(shape):
    tf.set_random_seed(1)
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(z):
    return tf.nn.max_pool(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def cnn_model(X_train, y_train, keep_prob=0.8, lamda=1e-4, num_epochs=450):
    print('X_train shape:',X_train.shape)
    print('y_train shape:',y_train.shape)
    X = tf.placeholder(tf.float32, [None, 64, 64, 3], name='input_x')
    y = tf.placeholder(tf.float32, [None, global_vals.num_classes], name='input_y')
    kp = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
    lam = tf.placeholder(tf.float32, name='lambda')

    # conv1
    # input: [None,64,64,3]
    # output: [None,32,32,32]
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    maxpool1 = max_pool_2x2(z1)

    # conv2
    # output: [None,16,16,64]
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)
    maxpool2 = max_pool_2x2(z2)

    # full connection1
    # output: [None,200]
    W_fc1 = weight_variable([16 * 16 * 64, 200])
    b_fc1 = bias_variable([200])
    maxpool2_flat = tf.reshape(maxpool2, [-1, 16 * 16 * 64])
    z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1,name='output_vector')
    z_fc1_drop = tf.nn.dropout(z_fc1, keep_prob=kp)

    # softmax layer
    # output: [None,num_classes]
    W_fc2 = weight_variable([200,global_vals.num_classes])
    b_fc2 = bias_variable([global_vals.num_classes])
    z_fc2 = tf.add(tf.matmul(z_fc1_drop, W_fc2), b_fc2, name='outlayer')
    prob = tf.nn.softmax(z_fc2, name='probability')

    # cost function
    regularizer = tf.contrib.layers.l2_regularizer(lam)
    regularization = regularizer(W_fc1) + regularizer(W_fc2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)) + regularization

    train = tf.train.AdamOptimizer().minimize(cost)
    # output_type='int32', name="predict"
    pred = tf.argmax(prob, 1, output_type='int32', name='predict')  # 输出结点名称predict方便后面保存为pb文件
    correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.set_random_seed(2018)  # to keep consistent results

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            train_generator=data_utils.data_augmentation()
            minibatch_cost=None
            batches=0
            for X_data,y_data in train_generator.flow(X_train,y_train):
                _,minibatch_cost=sess.run([train,cost],feed_dict={X:X_data,y:y_data,kp:keep_prob,lam:lamda})
                batches+=1
                if batches>=X_train.shape[0]: # that is 32 duplicates per image
                    break
            if epoch%10==0:
                print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
                print('cost after epoch {} :{}'.format(epoch,minibatch_cost))

        # 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
        train_acc = accuracy.eval(feed_dict={X: X_train[:100], y: y_train[:100], kp: 0.8, lam: lamda})
        print('train accuracy', train_acc)

        # save model
        saver = tf.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                                'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
        if not os.path.exists('model'):
            os.mkdir('model')
        saver.save(sess, os.path.join('model','cnn_model.ckpt'))
        # 将训练好的模型保存为.pb文件，方便在Android studio中使用
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                     output_node_names=['predict'])
        with tf.gfile.FastGFile(os.path.join('model','gesture.pb'),
                                mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    ### cnn model
    # print('载入数据集: ' + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    # X_train, y_train = data_utils.load_dataset()
    # print('开始训练: ' + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    # cnn_model(X_train, y_train, keep_prob=0.8, lamda=1e-4, num_epochs=1)
    # print('训练结束: ' + str((time.strftime('%Y-%m-%d %H:%M:%S'))))

    ### capsule network
    print('载入数据集: ' + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    X_train, y_train = data_utils.load_dataset()
    print('开始训练: ' + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    caps=capsule(batch_size=32)
    caps.train_and_save(X_train,y_train,lamda=1e-4,num_epochs=1)
    print('训练结束: ' + str((time.strftime('%Y-%m-%d %H:%M:%S'))))