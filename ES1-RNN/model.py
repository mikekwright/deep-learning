import tensorflow as tf
import numpy as np
import json

from pprint import pprint
from tensorflow.contrib import rnn
from tensorflow import nn


class RNNModel:
    ONE_HOT_SIZE = 63

    def __init__(self):
        self._test_data = None
        self._graph = None
        self._text = None
        self._target = None

    def build(self, hidden_n=128):
        # The next lines setup the graph, but we aren't required to do this on a fresh run
        #   useful if you are in jupyter
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Shape (None,   - The batch size (determine at runtime num of sequences)
            #        None,   - The max sequence size (determine at runtime length of a sequence) - sequence is a hashtag in our case
            #        63)     - One hot encoding vector size for a character in a sequence
            self._text = tf.placeholder(tf.float32, name='hashtags', shape=(None, None, RNNModel.ONE_HOT_SIZE))

            # Shape (None,   - The batch size (determine at runtime num of sequences)
            #        None,   - The sequence count
            self._target = tf.placeholder(tf.float32, name='encodings', shape=(None, None))

            # Holder for the sequence lengths
            self._q_len = tf.placeholder(tf.float32, name='sequence-length', shape=(None,))

            # This is the place for the RNN to start to take shape
            self._lstm_cell = rnn.BasicLSTMCell(hidden_n)

            #
            # Shape (None,   - The batch size (determine at runtime num of sequences)
            #        None,   - The max sequence size (determine at runtime length of a sequence) - sequence is a hashtag in our case
            #        128)    - This hidden layer size for the cell
            #
            self._rnn_output, self._rnn_states = nn.dynamic_rnn(self._lstm_cell, self._text, dtype=tf.float32, sequence_length=self._q_len)

            #
            #  Do a weight multiplication to get from shape
            #    (batch, length, one_hot) to (batch, length, 1)
            #

            weights_init = tf.contrib.layers.xavier_initializer()
            self._flattened_rnn = tf.reshape(self._rnn_output, [-1, hidden_n])
            self._output_weights = tf.get_variable(name='output-weights', dtype=tf.float32, shape=(hidden_n, 1), initializer=weights_init)

            bias_init = tf.constant_initializer(0)
            self._bias = tf.get_variable(name='output-bias', dtype=tf.float32, shape=(1,), initializer=bias_init)
            self._wx = tf.matmul(self._flattened_rnn, self._output_weights)
            # self._prediction = self._wx

            bias_add = nn.bias_add(self._wx, self._bias)
            text_shape = tf.shape(self._text)
            self._prediction = tf.reshape(bias_add, shape=(text_shape[0], text_shape[1]))

            # self._prediction = tf.matmul(self._wx, self._bias)

            # self._no_op = self._text * 1
            # self._no_op2 = self._target * 1

            self._initializer = tf.global_variables_initializer()

    def train(self, epochs=1, batch_size=5):
        self.build()

        sess = tf.Session(graph=self._graph)
        sess.run(self._initializer)
        test_set = self._test_set()

        start = 0
        end = batch_size
        for epoch in range(epochs):
            print(f'Running epoch {epoch} of {epochs}')
            pprint(test_set[start:end])

            max_sequence = max([len(b['phrase']) for b in test_set[start:end]])
            feed_dict = {
                self._text: [self._one_hot_encode(x['phrase'], max_length=max_sequence) for x in test_set[start:end]],
                self._target: [self._pad_word(x['normal'], max_length=max_sequence) for x in test_set[start:end]],
                self._q_len: [len(b['phrase']) for b in test_set[start:end]]
            }

            # print(sess.run([self._no_op, self._no_op2, self._q_len], feed_dict))
            # print(sess.run([self._rnn_output], feed_dict=feed_dict))

            # rnn_output = sess.run(self._rnn_output, feed_dict=feed_dict)
            # print(rnn_output.shape)

            output = sess.run(self._prediction, feed_dict=feed_dict)
            print(output.shape)

            start = end
            end = end + batch_size

    def _pad_word(self, word, max_length=None):
        update = list(word)
        pad_size = max_length - len(word)
        update.extend([0] * pad_size)
        return update

    def _test_set(self):
        if not self._test_data:
            self._load_test_data(limit=1000)

        np.random.shuffle(self._test_data)
        return self._test_data

    def _load_test_data(self, path='/data/twitter/testset.txt', limit=None):
        count = 0
        self._test_data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if limit and count > limit:
                    break
                count += 1
                self._test_data.append(json.loads(line))

    def _one_hot_encode(self, sequence, max_length=None):
        encoding = [self._character_to_one_hot(c) for c in sequence]
        if max_length and max_length > len(encoding):
            pad_size = max_length - len(encoding)
            encoding.extend([[0] * RNNModel.ONE_HOT_SIZE for _ in range(pad_size)])

        return encoding

    def _character_to_one_hot(self, c):
        index = -1
        c_ord = ord(c)
        if c.isdigit():
            index = int(c)
        elif c_ord >= ord('a') and c_ord <= ord('z'):
            index = (c_ord - ord('a')) + 10
        elif c_ord >= ord('A') and c_ord <= ord('Z'):
            index = (c_ord - ord('A')) + 36

        encoding = np.zeros(RNNModel.ONE_HOT_SIZE)
        encoding[index] = 1
        return encoding
