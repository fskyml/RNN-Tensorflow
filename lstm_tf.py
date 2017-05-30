"""
We define the structure of the LSTM network.
"""
import tensorflow as tf


class LSTM(object):
    """
    Contains methods to initialize the LSTM network
    """
    # Directory where tensorboard data will be stored.
    LOG_DIRECTORY = '/home/arko/Documents/Python/Deep Learning/RecurrentNeuralNetwork/Tensorboard/LSTM'

    def __init__(self, state_size, num_classes, vocab_size, hidden_size=500, model_name='LSTM1'):
        """Initializes the LSTM network"""
        self.BATCH_SIZE = 256
        self.CHECKPOINT_PATH = '/home/arko/Documents/Python/Deep ' \
                               'Learning/RecurrentNeuralNetwork/Tensorboard/LSTM/checkpoints '
        self.model_name = model_name
        self.state_size = state_size
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        with tf.name_scope('Weights_Biases'):
            # Weight matrices.
            self.w = tf.get_variable(
                'W',
                shape=[4, self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )

            self.u = tf.get_variable(
                'U',
                shape=[4, self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )

            self.v = tf.get_variable(
                'V',
                shape=[4, self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )

            self.biases = tf.constant(
                0.1,
                shape=[1, self.vocab_size, self.hidden_size],
                name='Biases'
            )

            tf.summary.histogram('W', self.w)
            tf.summary.histogram('Biases', self.biases)
            tf.summary.histogram('U', self.u)
            tf.summary.histogram('V', self.v)

        def define_placeholders():
            """
            Defines the placeholders for the LSTM network.
            :return:
            """
            self.input_placeholder = tf.placeholder(
                shape=[None, self.vocab_size],
                dtype=tf.float32,
                name='Input'
            )

            self.output_placeholder = tf.placeholder(
                shape=[None, vocab_size],
                dtype=tf.float32,
                name='Output'
            )

            self.init_state_placeholder = tf.placeholder(
                shape=[1, self.hidden_size],
                dtype=tf.float32,
                name='Initial'
            )

        define_placeholders()

    def step(self, prev, current):
        """
        Operation to perform while scanning
        :param prev:
        :param current:
        :return:
        """
        ht_1, ct_1 = tf.unstack(prev)

        # Forget gate
        forget_gate = tf.sigmoid(
            tf.matmul(
                current, self.w[0]
            ) + tf.matmul(
                ht_1, self.u[0]
            ) + self.biases
        )

        # Input gate
        input_gate = tf.sigmoid(
            tf.matmul(
                current, self.w[1]
            ) + tf.matmul(
                ht_1, self.u[1]
            ) + self.biases
        )

        # Candidate gate
        candidate_gate = tf.tanh(
            tf.matmul(
                current, self.w[2]
            ) + tf.matmul(
                ht_1, self.u[2]
            ) + self.biases
        )

        # Output gate
        output_gate = tf.sigmoid(
            tf.matmul(
            current, self.w[3])
        ) + tf.matmul(
            ht_1, self.u[3]
        ) + tf.matmul(
            candidate_gate, self.v[3]
        ) + self.biases
        

        ct = (ct_1 * forget_gate) + (candidate_gate * input_gate)

        ht = tf.tanh(ct) * output_gate

        return tf.stack([ht, ct])
