"""
We define the structure of the LSTM network.
"""
import os

import numpy
import tensorflow as tf


class LSTM(object):
    """
    Contains methods to initialize the LSTM network
    """
    # Directory where tensorboard data will be stored.
    LOG_DIRECTORY = '/home/arko/Documents/Python/Deep Learning/RecurrentNeuralNetwork/Tensorboard/LSTM'

    def __init__(self, vocab_size, hidden_size=500, model_name='LSTM1'):
        """Initializes the LSTM network"""
        tf.reset_default_graph()
        print('Building graph!')
        self.BATCH_SIZE = 256
        self.CHECKPOINT_PATH = '/home/arko/Documents/Python/Deep ' \
                               'Learning/RecurrentNeuralNetwork/Tensorboard/LSTM/checkpoints '
        self.model_name = model_name
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
                shape=[4, self.hidden_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )

            self.v = tf.get_variable(
                'V',
                shape=[4, self.hidden_size, self.hidden_size],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )

            self.biases = tf.constant(
                0.1,
                shape=[self.vocab_size, self.hidden_size],
                name='Biases'
            )

            # Add the distributions to the tensorboard.
            tf.summary.histogram('W', self.w)
            tf.summary.histogram('Biases', self.biases)
            tf.summary.histogram('U', self.u)
            tf.summary.histogram('V', self.v)

        self.input_placeholder = tf.placeholder(
            shape=[None, None],
            dtype=tf.float32,
            name='Input'
        )

        self.output_placeholder = tf.placeholder(
            shape=[None],
            dtype=tf.float32,
            name='Output'
        )

        self.init_state_placeholder = tf.placeholder(
            shape=[2, None, self.hidden_size],
            dtype=tf.float32,
            name='Initial'
        )

        self.processed_input = self.process_input(self.input_placeholder)

    def lstm_step(self, previous_hidden_memory, current_input):
        """
        Outputs the current hidden state using the 2 parameters. This is meant to be used with the tf.scan method.
        :param previous_hidden_memory: 
        :param current_input: 
        :return: 
        """
        previous_hidden_state, previous_memory_cell = tf.unstack(previous_hidden_memory)

        # We have assigned the following convention: for the weight matrices w,u,v
        # weight[0]: forget gate,
        # weight[1]: input gate,
        # weight[2]: candidate value gate,
        # weight[3]: output gate

        # Input gate
        input_gate = tf.sigmoid(
            tf.matmul(
                current_input, self.w[1]
            ) + tf.matmul(
                previous_hidden_state, self.u[1]
            )
        )

        # Forget gate
        forget_gate = tf.sigmoid(
            tf.matmul(
                current_input, self.w[0]
            ) + tf.matmul(
                previous_hidden_state, self.u[0]
            )
        )

        # Candidate gate
        c_ = tf.tanh(
            tf.matmul(
                current_input, self.w[2]
            ) + tf.matmul(
                previous_hidden_state, self.u[2]
            )
        )

        current_memory_cell = (input_gate * c_) + (forget_gate * previous_memory_cell)

        # Output gate
        output_gate = tf.sigmoid(
            tf.matmul(
                current_input, self.w[3]
            ) + tf.matmul(
                previous_hidden_state, self.u[3]
            ) + tf.matmul(
                current_memory_cell, self.v[3]
            )
        )

        # Current hidden state
        current_hidden_state = output_gate * tf.tanh(current_memory_cell)

        return tf.stack([current_hidden_state, current_memory_cell])

    def get_states(self):
        """
        Returns a list of states and builds the computation graph.
        :return: 
        """
        initiator = tf.constant(
            0.0,
            shape=[2, 1, self.hidden_size]
        )

        hidden_state_list = tf.scan(
            self.lstm_step,
            self.processed_input,
            initializer=initiator,
            name='States'
        )
        return hidden_state_list

    def train(self, data, character_to_index):
        """
        Train the network with the help of the computational graph.
        :return: 
        """
        states = self.get_states()

        with tf.name_scope('Cross-Entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.output_placeholder, logits=states
            )
            tf.summary.scalar('CrossEntropy', cross_entropy)

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('Loss', loss)

        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        with tf.name_scope('Accuracy'):
            correct = tf.equal(
                tf.argmax(states, 1), tf.argmax(self.output_placeholder, 1)
            )

            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('Accuracy', accuracy)

        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            # Initialize all variables.
            sess.run(tf.global_variables_initializer())
            # Variables for tensorboard.
            file_writer = tf.summary.FileWriter(self.LOG_DIRECTORY)
            file_writer.add_graph(sess.graph)
            saver = tf.train.Saver()

            # Training step
            i = 0
            while i + self.BATCH_SIZE < len(data):

                input_value = [character_to_index[ch] for ch in data[i: i + self.BATCH_SIZE]]

                target_value = [character_to_index[ch] for ch in data[i + 1: i + self.BATCH_SIZE + 1]]

                # One hot encode these values!
                input_value = numpy.eye(self.vocab_size)[input_value]
                target_value = numpy.eye(self.vocab_size)[target_value]

                train_loss, _ = sess.run(
                    [loss, optimizer],
                    feed_dict={
                        self.input_placeholder: input_value,
                        self.output_placeholder: target_value,
                        self.init_state_placeholder: numpy.zeros([2, self.BATCH_SIZE, self.hidden_size])
                    }
                )
                print(i)
                if i % 5 == 0:
                    [_, s] = sess.run(
                        [accuracy, summary],
                        feed_dict={
                            self.input_placeholder: input_value,
                            self.output_placeholder: target_value,
                            self.init_state_placeholder: numpy.zeros([2, self.BATCH_SIZE, self.hidden_size])
                        }
                    )
                    file_writer.add_summary(s, i)

                if i % 100 == 0:
                    saver.save(sess, os.path.join(self.LOG_DIRECTORY, 'model.ckpt'), i)

                i += self.BATCH_SIZE

    def process_input(self, input_placeholder):
        batch_input = tf.transpose(
            input_placeholder,
            perm=[2, 0, 1]
        )
        return tf.transpose(batch_input)
