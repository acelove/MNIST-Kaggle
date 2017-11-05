#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from utils import *
import math
class CNN(object):
    def __init__(self,num_of_labels):
        self.X = tf.placeholder(tf.float32,[None,784])
        self.y_ = tf.placeholder(tf.int32,[None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_conv = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        y_one_hot = tf.one_hot(self.y_,depth=num_of_labels,on_value=1,off_value=0)

        x_image = tf.reshape(self.X,[-1,28,28,1])

        #第一层卷积
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([6,6,1,24])
            b_conv1 = bias_variable([24])
            h_conv1 = conv2d(x_image,W_conv1)
            h_bn_conv1 = batchnorm(h_conv1,b_conv1,convolutional=True)
            h_act_conv1 = tf.nn.relu(h_bn_conv1)
            h_drop_conv1 = tf.nn.dropout(h_act_conv1,self.keep_prob_conv,compatible_convolutional_noise_shape(h_act_conv1))


        #第二层卷积
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5,5,24,48])
            b_conv2 = bias_variable([48])
            h_conv2 = conv2d(h_drop_conv1,W_conv2)
            h_pool_conv2 = max_pool_2x2(h_conv2)
            h_bn_conv2 = batchnorm(h_pool_conv2,b_conv2,convolutional=True)
            h_act_conv2 = tf.nn.relu(h_bn_conv2)
            h_drop_conv2 = tf.nn.dropout(h_act_conv2,self.keep_prob_conv, compatible_convolutional_noise_shape(h_act_conv2)) 


        #第三层卷积
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([4,4,48,64])
            b_conv3 = bias_variable([64])
            h_conv3 = conv2d(h_drop_conv2,W_conv3)
            h_pool_conv3 = max_pool_2x2(h_conv3)
            h_bn_conv3 = batchnorm(h_pool_conv3,b_conv3,convolutional=True)
            h_act_conv3 = tf.nn.relu(h_bn_conv3)
            h_drop_conv3 = tf.nn.dropout(h_act_conv3,self.keep_prob_conv,compatible_convolutional_noise_shape(h_act_conv3))


        #第一层全连接
        with tf.variable_scope('fc1'):
            W_fc1 = weight_variable([7*7*64,256])
            b_fc1 = bias_variable([256])
            h_fc1 = tf.matmul(tf.reshape(h_drop_conv3,[-1,7*7*64]),W_fc1)
            h_bn_fc1 = batchnorm(h_fc1,b_fc1)
            h_act_fc1 = tf.nn.relu(h_bn_fc1)
            h_drop_fc1 = tf.nn.dropout(h_act_fc1,self.keep_prob)


        #第二层全连接
        with tf.variable_scope('fc2'):
            W_fc2 = weight_variable([256,num_of_labels])
            b_fc2 = bias_variable([num_of_labels])
            h_fc2 = tf.matmul(h_drop_fc1,W_fc2)+b_fc2


        self.predict = tf.argmax(tf.nn.softmax(h_fc2),1)
        correct_prediction = tf.cast(tf.equal(self.predict, tf.argmax(y_one_hot, 1)),tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot,logits=h_fc2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self,sess,batch_X,batch_y,iter,keep_prob,keep_prob_conv):
        max_learning_rate = 0.02
        min_learning_rate = 0.0001
        decay_speed = 1600
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-iter/decay_speed)  
        sess.run(self.optimizer,
                 feed_dict={
                     self.X:batch_X,
                     self.y_:batch_y,
                     self.keep_prob:keep_prob,
                     self.keep_prob_conv:keep_prob_conv,
                     self.lr:learning_rate
                 }
                )

    def get_loss(self,sess,train_X,train_y):
        return sess.run(self.loss,
                        feed_dict={
                            self.X:train_X,
                            self.y_:train_y,
                            self.keep_prob:1.0,
                            self.keep_prob_conv:1.0
                        }
                       )

    def get_predict(self,sess,test_X):
        return sess.run(self.predict,
                        feed_dict={
                            self.X:test_X,
                            self.keep_prob:1.0,
                            self.keep_prob_conv:1.0
                        }
                       )
    
    def get_accuracy(self,sess,test_X,test_y):
        return sess.run(self.accuracy,
                        feed_dict={
                            self.X:test_X,
                            self.y_:test_y,
                            self.keep_prob:1.0,
                            self.keep_prob_conv:1.0
                        }
                       )
