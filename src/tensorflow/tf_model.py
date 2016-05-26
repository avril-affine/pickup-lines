import tensorflow as tf
import numpy as np
from tf_layers import *


if __name__ == '__main__':
    X_data = np.random.normal(scale=0.1, size=(1005, 10)).astype(np.float32)
    y_data = np.random.randint(0, 2, size=(1005, 1)).astype(np.int32)
    y_data = np.hstack((y_data, -(y_data - 1)))
    X = tf.placeholder(tf.float32, [None, 10], 'X')
    y = tf.placeholder(tf.int32, [None, 2], 'y')

    #global_step = tf.Variable(0, name='global_step', trainable=False)

    fc1 = fc_layer(X, 10, 10, 'fc1')
    fc2 = fc_layer(fc1, 10, 2, 'fc2')
    preds = tf.argmax(fc2, 1)
    loss = softmax_loss(fc2, y, 'loss')

    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(.03).minimize(loss)
        #train_step = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

    sess = tf.Session()

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('/tmp/train', sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)

    batch_size = 100
    step = 1
    for i in xrange(50):
        for j in xrange(0, len(X_data), batch_size):
            feed_dict = {X: X_data[i:i+batch_size], y: y_data[i:i+batch_size]}
            sess.run(train_step, feed_dict=feed_dict)
            summary = sess.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            print sess.run(preds, feed_dict=feed_dict)
            #train_writer.add_summary(summary, global_step=global_step)
            step+=1

    train_writer.close()
