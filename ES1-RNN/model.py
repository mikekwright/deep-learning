import tensorflow as tf
import numpy as np
import json

from pprint import pprint
from tensorflow.contrib import rnn
from tensorflow import train
from tensorflow import nn

from tester import TEST_SET, validate_model, model_results


class RNNModel:
    ONE_HOT_SIZE = 64
    LAYER_SIZE=128
    LEARNING_RATE=1e-3
    BATCH_SIZE=64
    EPOCHS=5
    CUTOFF=.5

    def __init__(self):
        self._test_data = None
        self._graph = None
        self._text = None
        self._target = None

    def build(self, input_n=LAYER_SIZE, learning_rate=LEARNING_RATE):
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
            self._q_len = tf.placeholder(tf.int32, name='sequence-length', shape=(None,))

            # This is the place for the RNN to start to take shape
            # self._lstm_cell = rnn.BasicLSTMCell(input_n)

            #
            # Shape (None,   - The batch size (determine at runtime num of sequences)
            #        None,   - The max sequence size (determine at runtime length of a sequence) - sequence is a hashtag in our case
            #        128)    - This hidden layer size for the cell
            #
            # self._rnn_output, self._rnn_states = nn.dynamic_rnn(self._lstm_cell, self._text, dtype=tf.float32, sequence_length=self._q_len)
            # hidden_n = input_n


            #
            # A Bi-directional RNN
            #
            self._forward_lstm = rnn.BasicLSTMCell(input_n)
            self._backward_lstm = rnn.BasicLSTMCell(input_n)
            bidir_output, self._rnn_states = nn.bidirectional_dynamic_rnn(self._forward_lstm, self._backward_lstm, self._text,
                dtype=tf.float32, sequence_length=self._q_len)
            self._rnn_output = tf.concat(bidir_output, axis=2)
            hidden_n = input_n * 2

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
            self._logits = tf.reshape(bias_add, shape=(text_shape[0], text_shape[1]))

            self._prediction = tf.sigmoid(self._logits)

            # sigmoid_crossy_entropy loss with logits
            # sigmoid = tf.sigmoid(bias_add)

            # self._prediction = tf.reshape(bias_add, shape=(text_shape[0], text_shape[1]))
            # self._prediction = tf.reshape(sigmoid, shape=(text_shape[0], text_shape[1]))

            # diff = tf.subtract(self._prediction, self._target)
            # losses = tf.abs(diff)
            # self._mask = tf.sequence_mask(self._q_len, dtype=tf.float32)
            # self._losses = self._mask * losses

            # token_cnt = tf.cast(tf.reduce_sum(self._q_len), tf.float32)

            # self._loss = tf.reduce_sum(tf.abs(diff))
            # self._loss = tf.reduce_mean(self._losses) / token_cnt
            # self._loss = tf.reduce_sum(self._losses) / token_cnt


            #diff = tf.subtract(self._prediction, self._target)
            # losses = tf.abs(diff)
            losses = nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=self._target)
            self._mask = tf.sequence_mask(self._q_len, dtype=tf.float32)
            self._losses = self._mask * losses

            token_cnt = tf.cast(tf.reduce_sum(self._q_len), tf.float32)
            self._loss = tf.reduce_sum(self._losses) / token_cnt


            # train_opt = train.GradientDescentOptimizer(1e-1)
            train_opt = train.AdamOptimizer(learning_rate=learning_rate)
            # self._train_step = train_opt.minimize(self._loss)

            gradients, variables = zip(*train_opt.compute_gradients(self._loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self._train_step = train_opt.apply_gradients(zip(gradients, variables))

            # self._prediction = tf.matmul(self._wx, self._bias)

            # self._no_op = self._text * 1
            # self._no_op2 = self._target * 1

            self._initializer = tf.global_variables_initializer()

    def _debug(self, sess, feed_dict, input_set):
        print('Batch: ', input_set)
        print('Target: ', sess.run(self._target, feed_dict=feed_dict))
        print('Logits: ', sess.run(self._logits, feed_dict=feed_dict))
        print('Predictions: ', sess.run(self._prediction, feed_dict=feed_dict))
        print('Losses: ', sess.run(self._loss, feed_dict=feed_dict))
        print('Training: ', sess.run(self._train_step, feed_dict=feed_dict))

    # def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
    def train(self, epochs=1, batch_size=64):
        self.build()

        sess = tf.Session(graph=self._graph)
        sess.run(self._initializer)

        for epoch in range(epochs):
            print(f'Running epoch {epoch} of {epochs}')

            test_set = self._test_set()
            batch_count = int(len(test_set) / batch_size)
            # batch_count = batch_count if batch_count < 10 else 10

            start = 0
            end = batch_size
            for batch in range(batch_count):
                input_set = test_set[start:end]
                # input_set = [{'normal': [0, 0, 1, 0, 0, 0], 'phrase': 'TheMan'}]

                # pprint(test_set[start:end])
                max_sequence = max([len(b['phrase']) for b in input_set])
                feed_dict = {
                    self._text: [self._one_hot_encode(x['phrase'], max_length=max_sequence) for x in input_set],
                    self._target: [self._pad_word(x['normal'], max_length=max_sequence) for x in input_set],
                    self._q_len: [len(b['phrase']) for b in input_set],
                }

                # print(sess.run([self._no_op, self._no_op2, self._q_len], feed_dict))
                # print(sess.run([self._rnn_output], feed_dict=feed_dict))

                # rnn_output = sess.run(self._rnn_output, feed_dict=feed_dict)
                # print(rnn_output.shape)

                # output = sess.run(self._prediction, feed_dict=feed_dict)
                # print(output.shape)
                # self._debug(sess, feed_dict, input_set)
                # break

                results = sess.run([self._train_step, self._loss, self._prediction], feed_dict=feed_dict)
                if batch % 10 == 0:
                    print(f'Batch {batch} of {batch_count}')
                    print(f'Loss: {results[1]}')

                start = end
                end = end + batch_size

                if batch % 10 == 0:
                    self._validate(sess)

        self._validate(sess)

    def _validate(self, sess):
        def test_model(word):
            feed_dict = {
                self._text: [self._one_hot_encode(word)],
                self._q_len: [len(word)]
            }

            value = sess.run(self._prediction, feed_dict=feed_dict)
            return [int(np.round(v)) for v in value[0]]

        print(f'Model results: {validate_model(test_model)}')



    def _pad_word(self, word, max_length=None):
        update = list(word)
        pad_size = max_length - len(word)
        update.extend([0] * pad_size)
        return update

    def _test_set(self):
        if not self._test_data:
            self._load_test_data()

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
        elif c == '_':
            index = -2

        encoding = np.zeros(RNNModel.ONE_HOT_SIZE)
        encoding[index] = 1
        return encoding
