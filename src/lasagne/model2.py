from vectorizer import Vectorizer
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from lasagne.nonlinearities import rectify, softmax
from lasagne.updates import nesterov_momentum, adam


class SaveBestModel(object):
    """Class to save the model if the validation accuracy improves."""
    def __init__(self, name, vectorizer, log_file='log2.txt'):
        self.name = name
        self.vec = vectorizer
        self.log_file = log_file
        self.epoch = 0

    def __call__(self, nn, train_history):
        if self.epoch % 10 == 0:
            nn.save_params_to('models/{}_{}.pkl'.format(self.name, self.epoch))
            line = self.generate_line(nn)
            with open(self.log_file, 'a') as f:
                f.write('Epoch{}\n'.format(self.epoch))
                f.write(line + '\n\n')
        self.epoch += 1

    def generate_line(self, nn, max_length=500):
        line = ''
        pred_char = None
        for _ in xrange(max_length):
            X, _ = self.vec.transform([line], train=False)
            probs = nn.predict_proba(X)
            pred = np.argmax(probs[-1])
            pred_char = self.vec.inv_chars[pred]
            if pred_char == self.vec.end_tag:
                break
            line += pred_char
        with open('chars_' + self.log_file, 'w') as f:
            for i, p in enumerate(probs[-1]):
                f.write('Epoch{}\n'.format(self.epoch))
                f.write('{}: {}\n'.format(self.vec.inv_chars[i], p))
        return line


def build_net(vectorizer, batch_size=1024*10, r1_size=100):
    vocab_size = vectorizer.num_chars
    seq_length = vectorizer.seq_length
    net = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('r1', layers.LSTMLayer),
                    ('s1', layers.SliceLayer),
                    ('output', layers.DenseLayer)],
            
            input_shape=(None, 25, vocab_size),

            r1_num_units=r1_size,

            s1_indices=-1,
            s1_axis=1,

            output_num_units=vocab_size,
            output_nonlinearity=softmax,

            update=nesterov_momentum,
            update_learning_rate=0.1,
            update_momentum=0.9,
            # update=adam,
            # update_learning_rate=0.01,

            max_epochs=10000,
            
            on_epoch_finished=[SaveBestModel('rnn', vectorizer)],

            batch_iterator_train=BatchIterator(batch_size),

            train_split=TrainSplit(eval_size=0.0),

            regression=False,
            verbose=2
        )
    return net


if __name__ == '__main__':
    print 'Loading data...'
    with open('data/data.txt', 'r') as f:
        data = [line.strip() for line in f]
    vectorizer = Vectorizer(seq_length=25)
    print 'Fitting Vectorizer...'
    X, y = vectorizer.fit_transform(data)
    with open('vectorizer.pkl', 'w') as f:
        pickle.dump(vectorizer, f)
    print 'Training Model...'
    net = build_net(vectorizer)
    try:
        net.fit(X, y)
    except KeyboardInterrupt:
        pass
