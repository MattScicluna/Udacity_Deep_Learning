# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
#  add 'LD_LIBRARY_PATH=/usr/local/cuda/lib64', 'CUDA_HOME=usr/local/cuda' to Python environment variable
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

'''
First reload the data we generated in 1_notmnist.ipynb.
'''

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

'''
Reformat into a shape that's more adapted to the models we're going to train:
-data as a flat matrix.
-labels as float 1-hot encodings.
'''

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...] (One Hot encoding)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

'''
Problem 1
Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding
a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor t using
nn.l2_loss(t). The right amount of regularization should improve your validation / test accuracy.
'''


def get_logit(tf_train_dataset, weights_to_hidden, biases_w_h, hidden_to_label, biases_h_l, dropout_prob):
        hidden_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, weights_to_hidden) + biases_w_h),
                                     keep_prob=dropout_prob)
        return tf.matmul(hidden_layer, hidden_to_label) + biases_h_l


def run_model(l2_penalty, batch_size, num_relu, num_steps, Dropout_prob, learning_rate):
    #  Dropout_prob = 1 means no dropout in model
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_to_hidden = tf.Variable(tf.truncated_normal([image_size * image_size, num_relu]))
        biases_w_h = tf.Variable(tf.zeros([num_relu]))

        hidden_to_label = tf.Variable(tf.truncated_normal([num_relu, num_labels]))
        biases_h_l = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = get_logit(tf_train_dataset, weights_to_hidden, biases_w_h, hidden_to_label, biases_h_l, Dropout_prob)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        # Add l2 loss
        loss += l2_penalty*(tf.nn.l2_loss(weights_to_hidden) + tf.nn.l2_loss(biases_w_h)
                            + tf.nn.l2_loss(hidden_to_label) + tf.nn.l2_loss(biases_h_l))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            get_logit(tf_valid_dataset, weights_to_hidden, biases_w_h, hidden_to_label, biases_h_l, dropout_prob=1))
        test_prediction = tf.nn.softmax(
            get_logit(tf_test_dataset, weights_to_hidden, biases_w_h, hidden_to_label, biases_h_l, dropout_prob=1))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized variables for 1-hidden layer neural network")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

        #  Print first 100 weights
        fig = plt.figure()
        fig.suptitle("Features from Neural Network", fontsize=16, fontweight='bold')
        fig_gs = gs.GridSpec(10, 10)
        features = weights_to_hidden

        for i in range(10):
            for j in range(10):
                ax = fig.add_subplot(fig_gs[i, j])
                ax.imshow(session.run(tf.reshape(features[:, i*10+j], [image_size, image_size])))
                ax.set_axis_off()
        fig.savefig("plots/Model Plot")

#run_model(l2_penalty=1e-03, batch_size=128, num_relu=1024, num_steps=3001, Dropout_prob=1, learning_rate=0.5)

'''
Problem 2
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
'''

#run_model(l2_penalty=1e-03, batch_size=30, num_relu=1024, num_steps=3001, Dropout_prob=1, learning_rate=0.5)
#  Validation and Test accuracy are very low

'''
Problem 3
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during
training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides
nn.dropout() for that, but you have to make sure it's only inserted during training.
What happens to our extreme overfitting case?
'''

#run_model(l2_penalty=1e-03, batch_size=128, num_relu=1024, num_steps=3001, Dropout_prob=0.5, learning_rate=0.5)
#run_model(l2_penalty=1e-03, batch_size=30, num_relu=1024, num_steps=3001, Dropout_prob=0.5, learning_rate=0.5)

'''
Problem 4
Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network
is 97.1%. One avenue you can explore is to add multiple layers.
Another one is to use learning rate decay:
'''


def get_deep_logit(tf_dataset, image_to_hidden1, biases_i_h1, hidden1_to_hidden2, biases_h1_h2,
                   hidden2_to_label, biases_h2_l, Dropout_prob_l1, Dropout_prob_l2):

    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_dataset, image_to_hidden1) + biases_i_h1),
                                 keep_prob=Dropout_prob_l1)
    hidden_layer_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer, hidden1_to_hidden2) + biases_h1_h2),
                                   keep_prob=Dropout_prob_l2)
    return tf.matmul(hidden_layer_2, hidden2_to_label) + biases_h2_l


def run_deep_model(l2_penalty, batch_size, num_layer_1, num_layer_2, num_steps,
                   Dropout_prob_l1, Dropout_prob_l2, learning_rate=0.5, decay_steps=10000, decay_rate=0.95):

    #  Dropout_prob = 1 means no dropout in model
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=decay_rate)

        # Variables.
        image_to_hidden1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_layer_1]))
        biases_i_h1 = tf.Variable(tf.zeros([num_layer_1]))

        hidden1_to_hidden2 = tf.Variable(tf.truncated_normal([num_layer_1, num_layer_2]))
        biases_h1_h2 = tf.Variable(tf.zeros([num_layer_2]))

        hidden2_to_label = tf.Variable(tf.truncated_normal([num_layer_2, num_labels]))
        biases_h2_l = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = get_deep_logit(tf_train_dataset, image_to_hidden1, biases_i_h1, hidden1_to_hidden2, biases_h1_h2,
                                hidden2_to_label, biases_h2_l, Dropout_prob_l1, Dropout_prob_l2)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + \
               l2_penalty * (tf.nn.l2_loss(image_to_hidden1) + tf.nn.l2_loss(biases_i_h1) +
                             tf.nn.l2_loss(hidden1_to_hidden2) + tf.nn.l2_loss(biases_h1_h2) +
                             tf.nn.l2_loss(hidden2_to_label) + tf.nn.l2_loss(biases_h2_l))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            get_deep_logit(tf_valid_dataset, image_to_hidden1, biases_i_h1, hidden1_to_hidden2,
                      biases_h1_h2, hidden2_to_label, biases_h2_l, Dropout_prob_l1=1, Dropout_prob_l2=1))
        test_prediction = tf.nn.softmax(
            get_deep_logit(tf_test_dataset, image_to_hidden1, biases_i_h1, hidden1_to_hidden2,
                      biases_h1_h2, hidden2_to_label, biases_h2_l, Dropout_prob_l1=1, Dropout_prob_l2=1))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized variables for 2-hidden layer neural network")

        saver = tf.train.Saver()

        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                if step % 5000 == 0:
                    saver.save(session, 'models/Model_at_Step_{}.ckpt'.format(step))
                    print("Model saved in file: %s" % 'models/Model_at_Step_{}.ckpt'.format(step))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#run_deep_model(l2_penalty=1e-04, batch_size=500, num_layer_1=1024, num_layer_2=512,
#               num_steps=60001, learning_rate=5e-5, Dropout_prob_l1=0.8, Dropout_prob_l2=0.8)
