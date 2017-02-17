#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#  From Udacity Deep Learning course Quiz:Softmax


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = [1, 2, 3]
vals = np.zeros((90, 3))
n = 0
for i in np.arange(1, 10, 0.1):
    print(np.multiply(x, i))
    vals[n] = softmax(np.multiply(x, i))
    n += 1


x = np.arange(1, 10, 0.1)
plt.plot(x, vals, linewidth=2)
plt.show()

#  If you multiply the probabilities, probabilities get closer to 0 and 1!

x = [1, 2, 3]
vals = np.zeros((90, 3))
n = 0
for i in np.arange(1, 10, 0.1):
    print(np.add(x, i))
    vals[n] = softmax(np.add(x, i))
    n += 1

x = np.arange(1, 10, 0.1)
plt.plot(x, vals, linewidth=2)
plt.show()

#  If you add the probabilities by a constant there is no difference!

x = [1, 2, 3]
vals = np.zeros((90, 3))
n = 0
for i in np.arange(1, 10, 0.1):
    print(np.divide(x, i))
    vals[n] = softmax(np.divide(x, i))
    n += 1


x = np.arange(1, 10, 0.1)
plt.plot(x, vals, linewidth=2)
plt.show()

# If you divide by constant the probabilities get closer to uniform!

#  Numeric Stability

x = 1000000000

for i in range(1,1000000):
    x = x + 0.000001

x = x - 1000000000

#x  = 0.95!?

'''
#  Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#  Start Tensorflow interactive session
#  Normally in TensorFlow programs you first create a graph and then launch it in a session
#  With interactive session we can go out of order
sess = tf.InteractiveSession()

#  Start building computation graph, create nodes for input images and target output classes.
#  784 comes from the 28x28 pixel flattened MNIST image.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
'''