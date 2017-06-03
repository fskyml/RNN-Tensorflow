"""
We wish to create a model of the LSTM network
"""
import numpy as np
import tensorflow as tf


class LSTM(object):
    """
    Defines the model of the lstm network
    """
    # The variables that we define later in the class.
    # Placeholders.
    input_placeholder = None
    output_placeholder = None
    initial_state_placeholder = None
    loss = None
    optimizer = None

    # Weight matriced
    hidden_weight = None
    input_weight = None

    # Hyperparameters
    hidden_size = None
    batch_size = 256

    # Model parameters
    ckpt_path = None
    model_name = None

    fixing_tensor = None
    fixing_bias = None

    def __init__(
            self,
            hidden_size: int,
            batch_size: 256,
            ckpt_path='Tensorboard/LSTM/ckpt/LSTM1',
            model_name='lstm1'
    ):
        """
        Initializes an instance of the LSTM network
        """
        self.hidden_size = hidden_size
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.batch_size = batch_size

        self.initialize_placeholder()
        self.initialize_weights_biases()

    def initialize_placeholder(self):
        """
        Initializes the placeholders
        """
        self.input_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.batch_size],
            name='Input_Placeholder'
        )
        self.output_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.batch_size],
            name='Outuput_Placeholder'
        )

        self.initial_state_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[2, 1, self.hidden_size],
            name='Initial_state_placeholder'
        )

    def initialize_weights_biases(self):
        """Initializes the weights and biases for the network"""
        self.hidden_weight = tf.get_variable(
            name='hidden_weight',
            shape=[4, self.hidden_size, self.hidden_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer
        )

        self.input_weight = tf.get_variable(
            name='input_weight',
            shape=[4, self.batch_size, self.hidden_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer
        )

    def step(self, prev, current):
        """Each step to be performed by the scan operation"""

        previous_memory_cell, previous_hidden_state = tf.unstack(prev)
        # We define the gates in the lstm cell
        # We have assigned the following codes:
        # 0: Input gate
        # 1: Candidate gate
        # 2: Forget gate
        # 3: Output gate
        current = tf.reshape(current, shape=[1, self.batch_size])
        # Input gate
        input_gate = tf.sigmoid(
            tf.matmul(
                current, self.input_weight[0]
            ) + tf.matmul(
                previous_hidden_state, self.hidden_weight[0]
            )
        )

        # Candidate gate
        candidate_gate = tf.tanh(
            tf.matmul(
                current, self.input_weight[1]
            ) + tf.matmul(
                previous_hidden_state, self.hidden_weight[1]
            )
        )

        # Forget gate
        forget_gate = tf.sigmoid(
            tf.matmul(
                current, self.input_weight[2]
            ) + tf.matmul(
                previous_hidden_state, self.hidden_weight[2]
            )
        )

        # Output gate
        output_gate = tf.sigmoid(
            tf.matmul(
                current, self.input_weight[3]
            ) + tf.matmul(
                previous_hidden_state, self.hidden_weight[3]
            )
        )

        # New memory cell
        new_memory_cell = previous_memory_cell*forget_gate + candidate_gate*input_gate

        new_state = tf.tanh(new_memory_cell) * output_gate

        return tf.stack([new_memory_cell, new_state])

    def get_states(self):
        """Returns the states of the hidden list"""
        hidden_states = tf.scan(
            self.step,
            self.input_placeholder,
            self.initial_state_placeholder
        )
        return hidden_states

    def train(self, data: str, character2idx: dict, number_epox=100):
        """
        Trains the lstm model
        """
        # A tensor that is used to get the final data into shape.
        with tf.Session() as sess:
            self.fixing_tensor = tf.get_variable(
                name='v',
                shape=[(2*self.hidden_size), self.batch_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer
            )
            # And the bias.
            self.fixing_bias = tf.get_variable(
                name='FixingBias',
                shape=[1, self.batch_size],
                initializer=tf.constant_initializer(0.)
            )

            hidden_states = self.get_states()

            # Variables associated with the network.
            hidden_states_reshaped = tf.reshape(hidden_states, shape=[1, (2 * self.hidden_size)])
            logits = tf.matmul(
                hidden_states_reshaped, self.fixing_tensor
            ) + self.fixing_bias

            predictions = tf.nn.softmax(logits=logits)

            total_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.output_placeholder, logits=predictions
            )
            self.loss = tf.reduce_mean(total_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
                # Initialize all the global variables.
            sess.run(tf.global_variables_initializer())
            # lets try to understand how the graph will execute
            for epoch in range(number_epox):
                train_loss = 0
                i = 0
                init_state = np.random.rand(2, 1, self.hidden_size)
                while i + self.batch_size <= len(data):
                    # Prepare the data.
                    # These are vectors of order [1, batch_size]
                    input_values = [
                        character2idx[ch] for ch in data[i: i + self.batch_size]
                    ]
                    target_values = [
                        character2idx[ch] for ch in data[i + 1: i + self.batch_size + 1]
                    ]
                    # A little hack to get the target to the right way.
                    if i + self.batch_size + 1 >= len(data):
                        target_values.append(-1)
                    # another little hack to get the data into right dimension
                    input_values = np.asarray(input_values).reshape([1, self.batch_size])
                    target_values = np.asarray(target_values).reshape([1, self.batch_size])

                    batch_train_loss, _ = sess.run(
                        [self.loss, self.optimizer],
                        feed_dict={
                            self.input_placeholder: input_values,
                            self.output_placeholder: target_values,
                            self.initial_state_placeholder: init_state
                        }
                    )
                    train_loss += batch_train_loss
                    i += self.batch_size
                print('Epoch: ', epoch, '\t|Training loss: ', train_loss/100)
