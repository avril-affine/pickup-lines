import numpy as np


class Vectorizer(object):
    def __init__(self, seq_length, start_tag='<START>', end_tag='<END>'):
        self.seq_length = seq_length
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.num_chars = 0
        self.chars_dict = {}
        self.inv_chars = {}

    def fit(self, X):
        """Fits list of documents and stores its chars_dict, 
        inv_chars, and num_chars.

        input:
            X(list of strings): each element of list is an observation
        """

        # get chars dict
        chars_set = reduce(lambda x,y: set(x).union(set(y)), X)
        num_chars = len(chars_set)
        self.chars_dict = dict(zip(chars_set, range(num_chars)))
        self.inv_chars = dict(zip(range(num_chars), chars_set))
        self.chars_dict[self.start_tag] = num_chars
        self.chars_dict[self.end_tag] = num_chars + 1
        self.inv_chars[num_chars] = self.start_tag
        self.inv_chars[num_chars+1] = self.end_tag

        self.num_chars = len(self.chars_dict)

        return self

    def transform(self, X, train=True):
        """Transforms list of documents to a char vector. Raises error if
        object has not been fit yet.

        input:
            X(list of strings): each element of list is an observation

        output:
            X(ndarray): List of sequences (rows, seq_length, num_chars)
            y(ndarray): List of targets (rows, num_chars)
        """
        if self.num_chars == 0:
            raise Exception('Object has not been fitted')
        X = self.process_data(X)
        return self._transform(X, train)

    def _transform(self, X, train=True):
        """Transforms list of documents to a char vector.

        input:
            X(list of strings): each element of list is an observation
        
        output:
            X(ndarray): List of sequences (rows, seq_length, num_chars)
            y(ndarray): List of targets (rows, num_chars)
        """

        X = [self.place_markers(line, train) for line in X]

        # flatten data and create X and y
        X = [ch for line in X for ch in line]
        y = X[1:] + [self.start_tag]     # offset by 1 and add start_tag to end 

        # convert to vector
        X = [self.char_vector(ch) for ch in X]
        y = [self.chars_dict[ch] for ch in y]

        X, y = self.make_sequences(X, y)

        return X, y

    def fit_transform(self, X):
        """Fit and transform X"""
        X = self.process_data(X)
        self.fit(X)
        return self._transform(X)

    def process_data(self, data):
        """Transform string to list of list of chars

        input:
            data(list): List of strings where each element is an observation

        output:
            res(list of list): List of list of chars
        """
        # replace newlines and tabs
        data = [line.replace('\\n', '\n') for line in data]
        data = [line.replace('\\t', '\t') for line in data]

        # convert to chars
        data = [list(line) for line in data]
        
        return data

    def make_sequences(self, X, y):
        """Converts X and y to sequences

        input:
            X(list of list): List of char vectors representing input
            y(list of list): List of char vectors representing target

        output:
            X(ndarray): List of sequences (rows, seq_length, num_chars)
            y(ndarray): List of targets (rows, num_chars)
        """
        if len(X) - self.seq_length + 1 <= 0:
            return (np.array([np.array(X)]).astype(np.float32), 
                    np.array([np.array(y)]).astype(np.int32))
        new_X = []
        new_y = []

        for i in xrange(len(X) - self.seq_length + 1):
            new_X.append(X[i:i+self.seq_length])
            new_y.append(y[i+self.seq_length-1])

        return np.array(new_X).astype(np.float32), np.array(new_y).astype(np.int32)

    def place_markers(self, line, train):
        """Place start and end tags to the line"""
        if train:
            return [self.start_tag] + line + [self.end_tag]
        else:
            return [self.start_tag] + line

    def char_vector(self, ch):
        """Make a char vector from a character"""
        vec = np.zeros(len(self.chars_dict))
        vec[self.chars_dict[ch]] = 1
        return vec.astype(np.float32)


def test():
    print 'Testing---------------------'
    data = ['Hello World', 'Foo bar baz buzz']
    print data
    v = Vectorizer(5)
    X, y = v.fit_transform(data)
    print 'X:', X.shape
    print 'y:', y.shape

    xx = [v.inv_chars[np.argmax(ch)] for ch in X[0]]
    yy = v.inv_chars[np.argmax(y[0])]
    print 'first sequence:', xx, repr(yy)
    xx = [v.inv_chars[np.argmax(ch)] for ch in X[1]]
    yy = v.inv_chars[np.argmax(y[1])]
    print 'second sequence:', xx, repr(yy)
    xx = [v.inv_chars[np.argmax(ch)] for ch in X[-1]]
    yy = v.inv_chars[np.argmax(y[-1])]
    print 'last sequence:', xx, repr(yy)

    print 'Testing2------------------'
    data2 = ['bbooooo World']
    print data2
    X2, _ = v.transform(data2)
    print 'X2:', X2.shape

    print 'Testing Empty---------------'
    empty, _ = v.transform([''], train=False)
    print 'empty string:', empty


if __name__ == '__main__':
    test()
