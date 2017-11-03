#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf

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

def binary(mat):
  return (mat>128).astype(np.float32)
