#!/usr/bin/env python
# coding=utf-8
from model import *
import pandas
if __name__ == "__main__":
    test_X = normalize(pandas.read_csv('../input/test.csv').values)

    cnn = CNN(num_of_labels=10)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('../model/')  
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path) 

        prediction = cnn.get_predict(sess,test_X)
        result =pandas.DataFrame({'ImageId':np.array([i+1 for i in xrange(prediction.size)]),'Label':prediction.astype(np.int32)})
        result.to_csv('../output/output.csv',index=False)
