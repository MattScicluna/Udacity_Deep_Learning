#  Udacity Deep Learning Course Notes


#  Lecture 1: From Machine Learning to Deep Learning
import numpy as np
import matplotlib.pyplot as plt

#################################
####  10/31 Softmax Quiz 1  #####
#################################

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    return np.exp(x)/np.sum(np.exp(x), axis=0)

#################################
####  11/31 Softmax Quiz 2  #####
#################################


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


#########################################
####  18/31 Numeric Stability Quiz  #####
#########################################


x = 1000000000

for i in range(0, 1000000):
    x = x + 0.000001

x = x - 1000000000

# x  = 0.95!? But it should be 1!
# since 0.000001*1000000 = 1
# We want our variables to have 0 mean and equal variance when possible.
# For pixels, between [0, 255], subtract 128 and then divide by 128.
# Generate initial weights from Gaussian with zero mean and small sigma, so it has small peaks i.e. "isn't opinionated"
