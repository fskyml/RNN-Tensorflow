# We define the RNN Model using numpy.

import numpy as np


def softmax(x):
    """Compute the softmax values for each sets of values in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class RNNNumpy:
    """
    Defines the Recurrent Neural Network model using numpy
    """

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """Initializes a new instance of the Recurrent Neural Network using numpy."""

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Parameters that we wish to learn from the neural network U,V,W.
        # Order of U = 100 x 8000
        self.U = np.random.uniform(
            -np.sqrt(1.0 / word_dim),
            np.sqrt(1.0 / word_dim),
            (hidden_dim, word_dim)
        )

        # Order of V = 8000 x 100
        self.V = np.random.uniform(
            -np.sqrt(1.0 / hidden_dim),
            np.sqrt(1.0 / hidden_dim),
            (word_dim, hidden_dim)
        )

        # Order of W = 100 x 100
        self.W = np.random.uniform(
            -np.sqrt(1.0 / hidden_dim),
            np.sqrt(1.0 / hidden_dim),
            (hidden_dim, hidden_dim)
        )

    def forward_propagation(self, x):
        # The total number of time steps
        len_of_x = len(x)

        s = np.zeros((len_of_x + 1, self.hidden_dim))

        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((len_of_x, self.word_dim))

        for t in np.arange(len_of_x):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))

            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]

    def predict(self, x):
        """Performs forward propagation and returns the index of the highest score"""
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        l = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            l += -1 * np.sum(np.log(correct_word_predictions))
        return l

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        n = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / n

    def back_propagation_through_time(self, x, y):
        """Performs back propagation through time"""
        t = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)

        # Gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(t):
            dLdV += np.outer(delta_o[t], s[t])

            delta_t = self.V.dot(delta_o[t]) * (1 - (s[t] ** 2))

            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = self.W.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)

        return [dLdU, dLdV, dLdW]
