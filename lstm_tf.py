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

    def hidden_layer(self, name='Input Layer'):
        # defines the hidden layer for the LSTM network
        with tf.name_scope(name):
            weights = tf.get_variable(
                'weights',
                shape=[self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )

            biases = tf.get_variable(
                'biases',
                shape=[self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )
