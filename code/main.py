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
    train_X = normalize(train[:,1:])

    cnn = CNN(num_of_labels=10)
    epoch = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        count = 0
        end = 10001
        while True:
            epoch += 1
            seed = int(time.time())
            np.random.seed(seed)
            np.random.shuffle(train_X)
            np.random.seed(seed)
            np.random.shuffle(train_y)
            for i in xrange(0,train_X.shape[0],100):
                batch_X = train_X[i:i+100]
                batch_y = train_y[i:i+100]
                cnn.train(sess,batch_X,batch_y,count,0.5,1.0)
                count += 1
            print "count : %d,acc:%.6f." % (count,cnn.get_accuracy(sess,train_X[100:1000],train_y[100:1000]))
            saver.save(sess, "../model/" + 'model.ckpt', global_step=epoch)
            if count > end:
                break

if __name__ == "__main__":
    main()
