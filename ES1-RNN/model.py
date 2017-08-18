import tensorflow as tf
import numpy as np
import json


class RNNModel:
    ONE_HOT_SIZE = 63

    def __init__(self):
        self._test_data = None
        self._graph = None
        self._text = None
        self._target = None

    def build(self):
        # The next lines setup the graph, but we aren't required to do this on a fresh run
        #   useful if you are in jupyter
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Shape (None,   - The batch size (determine at runtime num of sequences)
            #        None,   - The sequence size (determine at runtime length of a sequence) - sequence is a hashtag in our case
            #        63)     - One hot encoding vector size for a character in a sequence
            self._text = tf.placeholder(tf.float32, name='hashtags', shape=(None, None, RNNModel.ONE_HOT_SIZE))

            # Shape (None,   - The batch size (determine at runtime num of sequences)
            #        None,   - The sequence size (determine at runtime length of a sequence) - sequence is a hashtag in our case
            #        1)      - The value of the character (0 to 1)
            self._target = tf.placeholder(tf.float32, name='encodings', shape=(None, None))

            self._no_op = self._text * 1
            self._no_op2 = self._target * 1

    def train(self, epochs=1, batch_size=10):
        self.build()

        sess = tf.Session(graph=self._graph)
        test_set = self._test_set()

        for epoch in range(epochs):
            print(f'Running epoch {epoch} of {epochs}')

            max_sequence = max([len(b['phrase']) for b in test_set[:batch_size]])
            feed_dict = {
                self._text: [self._one_hot_encode(x['phrase'], max_length=max_sequence) for x in test_set[:batch_size]],
                self._target: [self._pad_word(x['normal'], max_length=max_sequence) for x in test_set[:batch_size]],
            }

            print(sess.run([self._no_op, self._no_op2], feed_dict))

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
