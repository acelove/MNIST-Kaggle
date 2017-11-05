#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from utils import *
class CNN(object):
    def __init__(self,num_of_labels,keep_prob,learning_rate,istrain=False):
        self.X = tf.placeholder(tf.float32,[None,784])
        self.y_ = tf.placeholder(tf.int32,[None])

        y_one_hot = tf.one_hot(self.y_,depth=num_of_labels,on_value=1,off_value=0)

        x_image = tf.reshape(self.X,[-1,28,28,1])

        #第一层卷积
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([5,5,1,10])
            b_conv1 = bias_variable([10])
            h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

        with tf.variable_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)
 
        #第二层卷积
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5,5,10,20])
            b_conv2 = bias_variable([20])
            h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

        with tf.variable_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

        #第一层全连接
        with tf.variable_scope('fc1'):
            W_fc1 = weight_variable([7*7*20,256])
            b_fc1 = bias_variable([256])
            h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2,[-1,7*7*20]),W_fc1)+b_fc1)

        with tf.variable_scope('dropout'):
            h_fc1_drop = h_fc1
            if istrain:
                h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        #第二层全连接
        with tf.variable_scope('fc2'):
            W_fc2 = weight_variable([256,10])
            b_fc2 = bias_variable([10])
            h_fc2 = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

        self.predict = tf.argmax(tf.nn.softmax(h_fc2),1)
        correct_prediction = tf.cast(tf.equal(self.predict, tf.argmax(y_one_hot, 1)),tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot,logits=h_fc2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self,sess,train_X,train_y):
        sess.run(self.optimizer,feed_dict={self.X:train_X,self.y_:train_y})

    def get_loss(self,sess,train_X,train_y):
        return sess.run(self.loss,feed_dict={self.X:train_X,self.y_:train_y})

    def get_predict(self,sess,test_X):
        return sess.run(self.predict,feed_dict={self.X:test_X})
    
    def get_accuracy(self,sess,test_X,test_y):
        return sess.run(self.accuracy,feed_dict={self.X:test_X,self.y_:test_y})

