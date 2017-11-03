#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import pandas
import time
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.get_variable("weight",initializer=initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.get_variable("bias",initializer=initial)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def deepnn(x):
    x_image = tf.reshape(x,[-1,28,28,1])

    #第一层卷积
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

    with tf.variable_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
 
    #第二层卷积
    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

    with tf.variable_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    #第一层全连接
    with tf.variable_scope('fc1'):
        W_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2,[-1,7*7*64]),W_fc1)+b_fc1)

    with tf.variable_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #第二层全连接
    with tf.variable_scope('fc2'):
        W_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])
        h_fc2 = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

    return h_fc2,keep_prob

def main():
    train = pandas.read_csv('input/train.csv').values
    train_y = train[:,0]
    train_X = train[:,1:]

    X = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.int32,[None])
    y_one_hot = tf.one_hot(y_,depth=10,on_value=1,off_value=0)

    y,keep_prob = deepnn(X)

    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot,logits=y)
        loss_mean = tf.reduce_mean(loss)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_mean)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in xrange(2000):
            seed = int(time.time())
            np.random.seed(seed)
            np.random.shuffle(train_X)
            np.random.seed(seed)
            np.random.shuffle(train_y)
            for i in xrange(0,train_X.shape[0],50):
                batch_X = train_X[i:i+50]
                batch_y = train_y[i:i+50]
                train_step.run(feed_dict={X:batch_X,y_:batch_y,keep_prob:0.5})
                print sess.run(loss_mean,feed_dict={X:batch_X,y_:batch_y,keep_prob:0.5})

if __name__ == "__main__":
    main()
