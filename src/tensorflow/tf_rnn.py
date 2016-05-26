import time
import tensorflow as tf
import numpy as np
from tf_layers import *
from vectorizer import Vectorizer
import cPickle as pickle


def to_vector(y, size):
    N = y.shape[0]
    res = np.zeros((N, size))
    res[np.arange(N), y] = 1
    return res


def sample(sess, tensors, hidden_dim, vec, max_iter=50):
    res = ''
    initial_state = np.zeros((seq_length, 4 * hidden_dim))

    for _ in xrange(max_iter):
        X, y = vec.transform([res], False)
        feed_dict = {tensors['X']: X, tensors['initial_state']: initial_state}
        pred = sess.run(tensors['preds'], feed_dict=feed_dict)
        pred = vec.inv_chars[pred[-1]]
        if pred == vec.end_tag:
            break
        else:
            res += pred

        next_hidden = tensors['next_hidden'].eval(feed_dict=feed_dict)
        initial_state = np.vstack((initial_state, next_hidden))[1:]

    return res


if __name__ == '__main__':
    print 'Loading data...'
    with open('../../data/smalldata.txt', 'r') as f:
        data = [line.strip() for line in f]
    vectorizer = Vectorizer(seq_length=25)
    print 'Fitting Vectorizer...'
    X_data, y_data = vectorizer.fit_transform(data)

    with open('vectorizer.pkl', 'w') as f:
        pickle.dump(vectorizer, f)

    N, seq_length, input_dim = X_data.shape
    hidden_dim = 128
    output_dim = input_dim

    X = tf.placeholder(tf.float32, [None, seq_length, input_dim], 'X')
    y = tf.placeholder(tf.float32, [None, output_dim], 'y')
    initial_state = tf.placeholder(tf.float32, [None, 4 * hidden_dim], 'initial_state')
    
    lstm, next_hidden = lstm_layer(X, input_dim, seq_length, hidden_dim, 
                                   output_dim, initial_state, 'lstm')
    with tf.name_scope('predictions'):
        preds = tf.argmax(tf.nn.softmax(lstm), 1)
    loss = softmax_loss(lstm, y, 'loss')

    val_tensors = {'X': X,
                   'y': y,
                   'initial_state': initial_state,
                   'lstm': lstm,
                   'next_hidden': next_hidden,
                   'preds': preds}
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(.03).minimize(loss)

    sess = tf.Session()

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('/Users/PANDA/Dropbox/lstm/', sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=10)

    batch_size = 10280
    step = 1
    for i in xrange(1201):
        t = time.time()
        for j in xrange(0, len(X_data), batch_size):
            batch_x = X_data[i:i+batch_size]
            batch_y = y_data[i:i+batch_size]
            batch_y = to_vector(batch_y, output_dim)
            batch_initial = np.zeros((len(batch_x), 4 * hidden_dim))
            feed_dict = {X: batch_x, y: batch_y, initial_state:batch_initial}
            sess.run(train_step, feed_dict=feed_dict)
            summary = sess.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            step+=1
        print 'Epoch {}: finished in {}s'.format(i, time.time() - t)

        if i % 10 == 0:
            saver.save(sess, '/Users/PANDA/Dropbox/lstm/models/model.ckpt', global_step=i)
            with open('/Users/PANDA/Dropbox/lstm/models/tf_log.txt', 'a') as f:
                line = sample(sess, 
                              val_tensors,
                              hidden_dim, 
                              vectorizer)
                f.write('Epoch {}:\n{}\n'.format(i, line))

    train_writer.close()
