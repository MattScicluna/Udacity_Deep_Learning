#  Udacity Deep Learning Course Notes


#  Lecture 2: Deep Neural Networks


########################################
####  4/18 Rectified Linear Units  #####
########################################

#Rectified Linear Units are linear (y=x) if x>0 and 0 everywhere else.
#Simplest non-linear functions.

##########################
####  9/18 Backprop  #####
##########################

#Each block of the backprop takes about twice the memory as it does for the forward prop.
#Important for when you want to fit your model in memory.

#Increasing Hidden Layer size is inefficient for improving learning. Better to add more hidden layers (make model deeper).

################################
####  13/18 Regularization  ####
################################

#Hard to train model that is perfect size, so we train a larger model and use regularization to prevent overfitting.
#Ways to prevent overfitting: Early Stopping, Regularization, Dropout.

#########################
####  13/18 Dropout  ####
#########################

#Values from one layer to another called activations.
#Dropout -> Randomly, for half of the examples you train your network on, set half to zero, multiply the other half by 2.
#This is effectively destroying half of your data, and then randomly again.
#Forces network to learn redundant representations to ensure some of the information remains.
#Makes network more robust, as network acts like it is taking the consensus over an ensemble of networks.
#When evaluating models, want to take the average of the activations.

