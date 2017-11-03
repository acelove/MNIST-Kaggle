#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import pandas
import time
import numpy as np
from model import *

def main():
    train = pandas.read_csv('../input/train.csv').values
    train_y = train[:,0]
    train_X = binary(train[:,1:])

    cnn = CNN(10,0.5,1e-4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in xrange(2000):
            seed = int(time.time())
            np.random.seed(seed)
            np.random.shuffle(train_X)
            np.random.seed(seed)
            np.random.shuffle(train_y)
            for i in xrange(0,train_X.shape[0],50):
                batch_X = train_X[i:i+50]
                batch_y = train_y[i:i+50]
                cnn.train(sess,batch_X,batch_y)
                print cnn.get_loss(sess,batch_X,batch_y)
            saver.save(sess, "../model/" + 'model.ckpt', global_step=epoch+1)

if __name__ == "__main__":
    main()