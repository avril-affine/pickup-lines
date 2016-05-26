import tensorflow as tf


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def weights(shape, std=0.1):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial)


def bias(shape, value=0.0):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)


def fc_layer(input_tensor, input_dim, output_dim, name):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = weights([input_dim, output_dim])
            variable_summaries(W, name + '/weights')
        with tf.name_scope('bias'):
            b = bias([output_dim])
            variable_summaries(b, name + '/bias')
        with tf.name_scope('activations'):
            activations = tf.matmul(input_tensor, W) + b
            tf.histogram_summary(name + '/activations', activations)
        relu = tf.nn.relu(activations, 'relu')
        tf.histogram_summary(name + '/relu', relu)

    return relu


def lstm_layer(input_tensor, input_dim, seq_length, hidden_dim, output_dim, 
               initial_state, name, size=1, forget_bias=1.0):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W_in = weights([input_dim, hidden_dim])
            variable_summaries(W_in, name + '/weights_in')
        with tf.name_scope('bias'):
            b_in = bias([hidden_dim])
            variable_summaries(b_in, name + '/bias_in')

        input_tensor = tf.transpose(input_tensor, [1, 0, 2])
        input_tensor = tf.reshape(input_tensor, [-1, input_dim])
        input_tensor = tf.cast(input_tensor, tf.float32)
        with tf.name_scope('activations'):
            activations = tf.matmul(input_tensor, W_in) + b_in
            tf.histogram_summary(name + '/activations', activations)

        with tf.name_scope('lstm'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim,
                                                forget_bias=forget_bias)
            if size > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * size)

            activations = tf.split(0, seq_length, activations)

            lstm_out, lstm_state = tf.nn.rnn(lstm_cell, 
                                             activations,
                                             initial_state=initial_state,
                                             scope=name+'/lstm')
            
            with tf.name_scope('activations'):
                with tf.name_scope('weights'):
                    W_out = weights([hidden_dim, output_dim])
                    variable_summaries(W_out, name + '/weights_out')
                with tf.name_scope('bias'):
                    b_out = bias([output_dim])
                    variable_summaries(b_out, name + '/bias_out')
                out = tf.matmul(lstm_out[-1], W_out) + b_out

            tf.histogram_summary(name + '/lstm_last_activation', out)

    return out, lstm_state


def softmax_loss(scores, y_true, name):
    with tf.name_scope(name):
        probs = tf.nn.softmax(scores, 'probs')

        correct = tf.equal(tf.argmax(probs, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

        with tf.name_scope('cross_entropy'):
            y_true = tf.cast(y_true, tf.float32)
            loss = tf.nn.softmax_cross_entropy_with_logits(scores, y_true, 'loss')
            loss = tf.reduce_mean(loss)
        tf.scalar_summary('loss', loss)
    return loss
