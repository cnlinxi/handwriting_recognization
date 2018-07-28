# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 17:06
# @Author  : MengnanChen
# @FileName: capsule_network.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorflow.python.framework import graph_util
import time
import os

import capslayer
import global_vals
import data_utils


class capsule(object):
    def __init__(self, batch_size, num_classes=global_vals.num_classes):
        self.batch_size=batch_size
        self.num_classes = num_classes

    def train_and_save(self, X_train, y_train,num_epochs=420,lamda=1e-4):
        img_size=global_vals.resized_image_size
        num_classes=global_vals.num_classes
        X=tf.placeholder(tf.float32,[None,img_size,img_size,3],name='input_x')
        y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        lam=tf.placeholder(tf.float32,name='lambda')
        with tf.variable_scope('conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(X, num_outputs=256, kernel_size=9,
                                             stride=1, padding='VALID')

        with tf.variable_scope('primary_layer'):
            primary_caps, activation = capslayer.layers.primaryCaps(conv1, filters=32, kernel_size=9,
                                                                    strides=2, out_caps_shape=[8, 1],method='logistic')
        with tf.variable_scope('digit_layer'):
            primary_caps = tf.reshape(primary_caps, shape=[self.batch_size, -1, 8, 1])
            self.digit_caps, self.activation = capslayer.layers.fully_connected(primary_caps, activation,
                                                                                num_outputs=self.num_classes,
                                                                                out_caps_shape=[16, 1],
                                                                                routing_method='DynamicRouting')

        # input: [None,-1]
        # output: [None,global_vals.output_dim_vectors]
        dim_vectors=global_vals.output_dim_vectors
        W_fc = tf.Variable(tf.truncated_normal(shape=[self.activation.get_shape().as_list()[1], dim_vectors],
                                               stddev=0.1))
        b_fc=tf.Variable(tf.constant(0.0,shape=[dim_vectors]))
        z_fc=tf.nn.relu(tf.matmul(self.activation,W_fc)+b_fc,name='output_vector')

        # softmax layer
        # output: [None,num_classes]
        W_fc2=tf.Variable(tf.truncated_normal(shape=[dim_vectors,self.num_classes],
                                              stddev=0.1))
        b_fc2=tf.Variable(tf.constant(0.0,shape=[self.num_classes]))
        z_fc2=tf.nn.relu(tf.matmul(z_fc,W_fc2)+b_fc2,name='output_layer')
        prob=tf.nn.softmax(z_fc2,name='probability')

        # cost function
        regularizer=tf.contrib.layers.l2_regularizer(1e-4)
        regulazation=regularizer(W_fc)+regularizer(W_fc2)
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z_fc2))+regulazation

        train=tf.train.AdadeltaOptimizer().minimize(cost)
        pred=tf.argmax(prob,axis=1,output_type='int32',name='predict')
        correct_prediction=tf.equal(pred,tf.argmax(y,axis=1,output_type='int32'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.set_random_seed(2018)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                train_generator = data_utils.data_augmentation()
                minibatch_cost = None
                batches = 0
                for X_data, y_data in train_generator.flow(X_train, y_train):
                    _, minibatch_cost = sess.run([train, cost],
                                                 feed_dict={X: X_data, y: y_data, lam:lamda})
                    batches += 1
                    if batches >= X_train.shape[0]: # that is 32 duplicates per image
                        break
                if epoch % 10 == 0:
                    print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
                    print('cost after epoch {}:{}'.format(epoch, minibatch_cost))

            # 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
            train_acc = accuracy.eval(feed_dict={X: X_train[:100], y: y_train[:100], lam: lamda})
            print('train accuracy', train_acc)

            # save model
            saver = tf.train.Saver({'W_fc': W_fc, 'b_fc': b_fc, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
            if not os.path.exists('model'):
                os.mkdir('model')
            saver.save(sess, os.path.join('model', 'caps_model.ckpt'))
            # 将训练好的模型保存为.pb文件，方便在Android studio中使用
            output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                         output_node_names=['predict'])
            with tf.gfile.FastGFile(os.path.join('model', 'gesture_caps.pb'),
                                    mode='wb') as f: # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
                f.write(output_graph_def.SerializeToString())