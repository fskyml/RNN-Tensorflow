import os

import numpy as np
import tensorflow as tf

# Hyperparameters
hidden_size = 100
step_to_unroll = 100
learning_rate = 0.001

# Pre-processing!
# Data
data = open('/home/arko/Documents/Datasets/paulg/paulg.txt', 'r', encoding='utf8').read()
characters = sorted(  # Sorts the list for easy indexing. For some reason, the accuracy dips when it is not sorted!
    list(  # Creates a list out of it
        set(data)  # Extracts the unique elements.
    )
)

test_data = open('/home/arko/Documents/Datasets/paulg/paulg_test.txt', 'r').read()
test_character_index = {
    ch: i for i, ch in enumerate(sorted(list(set(test_data))))
}

test_index_character = {
    i: ch for i, ch in enumerate(sorted(list(set(test_data))))
}
vocab_size = len(characters)

# Used to convert integer to its equivalent character.
character_to_index = {
    ch: i for i, ch in enumerate(characters)  # Creates a dictionary with a character represented by an index.
}

index_to_character = {
    i: ch for i, ch in enumerate(characters)  # Creates a dictionary with each index representing a character.
}

# Placeholders.
# Ax81 matrix.
input_placeholder = tf.placeholder(
    shape=[None, vocab_size],
    dtype=tf.float32,
    name='Inputs'
)

# Ax81 matrix.
output_placeholder = tf.placeholder(
    shape=[None, vocab_size],
    dtype=tf.float32,
    name='Outputs'
)

# 1x100 matrix.
initial_state = tf.placeholder(
    shape=[1, hidden_size],
    dtype=tf.float32,
    name='InitialState'
)


# Define the initializer.
def random_normal_initialize():
    """
    Returns an initializer that can be used to initialize a tensor.
    :return: a random normal initializer
    """
    return tf.random_normal_initializer(stddev=0.1)


# Build the computation graph
with tf.variable_scope('ComputationGraph') as scope:
    hidden_state_at_t = initial_state
    output_state_list = []
    for t, input_state_at_t in enumerate(tf.split(input_placeholder, step_to_unroll, axis=0)):
        if t > 0:
            # Reuse the variables as this is the second time this loop is executing and the values should carry over.
            scope.reuse_variables()
        # The matrices that we wish to train and learn from the data.
        # 81x100 matrix
        W_xh = tf.get_variable(
            'W_xh',
            [vocab_size, hidden_size],
            initializer=random_normal_initialize()
        )
        tf.summary.histogram('WXH', W_xh)

        # 100x100 matrix.
        W_hh = tf.get_variable(
            'W_hh',
            [hidden_size, hidden_size],
            initializer=random_normal_initialize()
        )
        tf.summary.histogram('WHH', W_hh)

        # 100x81 matrix.
        W_yh = tf.get_variable(
            'W-Yh',
            [hidden_size, vocab_size],
            initializer=random_normal_initialize()
        )
        tf.summary.histogram('WYH', W_yh)

        # Update hidden state
        hidden_state_at_t = tf.tanh(
            tf.matmul(
                input_state_at_t,
                W_xh
            ) + tf.matmul(
                hidden_state_at_t,
                W_hh
            )
        )

        tf.summary.histogram('HiddenState', hidden_state_at_t)

        # Update the temporary output state.
        output_state_at_t = tf.matmul(
            hidden_state_at_t,
            W_yh
        )

        tf.summary.histogram('OutputState', output_state_at_t)

        # add this to the list so that we can backpropagate later.
        output_state_list.append(output_state_at_t)

previous_hidden_state = hidden_state_at_t
# Used for sampling later.
softmax = tf.nn.softmax(output_state_list[-1])

predicted_output = tf.concat(output_state_list, axis=0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits=predicted_output)
loss = tf.reduce_mean(cross_entropy)

tf.summary.scalar('CrossEntropy', cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    initiator = tf.global_variables_initializer()
    # Initialize all variables.
    sess.run(initiator)
    file_writer = tf.summary.FileWriter('/home/arko/Documents/Python/Deep '
                                        'Learning/RecurrentNeuralNetwork/Tensorboard/Vanilla RNN/hparam')
    file_writer.add_graph(sess.graph)
    p = 0
    previous_value_hidden = np.zeros([1, hidden_size])

    saver = tf.train.Saver()

    while p + step_to_unroll < len(data):
        # Prepare the inputs.
        input_value = [character_to_index[ch] for ch in data[p: p + step_to_unroll]]
        # We wish to predict the next character only.
        target_value = [character_to_index[ch] for ch in data[p + 1: p + step_to_unroll + 1]]

        # One Hot encode these values so that they can be fed to the dictionary.
        input_value = np.eye(vocab_size)[input_value]
        target_value = np.eye(vocab_size)[target_value]

        previous_value_hidden, _, cost = sess.run(
            [previous_hidden_state, optimizer, loss],
            feed_dict={
                input_placeholder: input_value,
                output_placeholder: target_value,
                initial_state: previous_value_hidden
            }
        )

        if p % 100 == 0:
            saver.save(sess, os.path.join('/home/arko/Documents/Python/Deep '
                                          'Learning/RecurrentNeuralNetwork/Tensorboard/Vanilla RNN/model.ckpt'), p)

        p += step_to_unroll
    correct = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(output_placeholder, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('Accuracy', accuracy)
    test_input = [test_character_index[ch] for ch in test_data[:1400]]
    test_output = [test_character_index[ch] for ch in test_data[1:1401]]

    # One hot encode these test values
    test_output = np.eye(vocab_size)[test_output]
    test_input = np.eye(vocab_size)[test_input]

    print('Accuracy: ', accuracy.eval({
        input_placeholder: test_input,
        output_placeholder: test_output,
        initial_state: np.zeros([1, hidden_size])
    }))
