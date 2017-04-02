#  Udacity Deep Learning Course Notes


#  Lecture 4: Deep Models for Text and Sequences


###########################
####  4/24 Embeddings  ####
###########################

#Want to predict a word's context, bringing similar words like "cat" and "kitty" closer together.
#Map words to small vectors, vectors closer together when words have similar meaning.

#########################
####  5/24 Word2Vec  ####
#########################

#Map word to a vector to try to predict words in context.
#By context here we mean words nearby in the same sentance. 
#Nearby words are predicted by Logistic Regression.
#Can project feature space into 2D space using tSNE.
#Measure closeness using cosine distance instead of L2, 
#since length of embedding vector is not relevant.
#Take word, embed it in a vector, apply linear model to the vector,
#Compare to target (words nearby in sentance), using sampled softmax.
#Sampled softmax = use only some of the words that are not a target, ignore rest.
#Reduces size of matrix and doesn't affect performance.

# e.g. 'cat' -> V('cat') -> Softmax{W*V('cat') + b} -> one-hot-encoding

###############################
####  9/24 Word Analogies  ####
###############################

#Semantic Analogy:
#Puppy -> Dog/Kitten -> Cat

#Syntactic Analogy:
#Taller -> Tall/Shorter -> Short

###################
#### 12/24 RNN ####
###################

#Sequence of events through time.
#If stationary, can use same classifier through time.
#Want to take into account the past, use state of previous classifier
# as a summary of what happened before, recursively.
#Network would be very deep though.
# We use tying to get a single model responsible for summarizing the past.

#####################################
#### 13/24 Backprop through time ####
#####################################

#Correlated parameter updates are bad for SGD.
# Causes gradients to blow up to infinity (Explode) or shrink to 0 (Vanish).

####################
#### 14/24 LSTM ####
####################

#Replace NN with LSTM 'Cell'
#Steps: 1. Write input X into memory M
#       2. Read it back to get Y
#       3. forget M.

#Gates that tell machine if it should do any of the 3 steps.
#These gates can be written as scalar 1 or 0 to multiply X, Y and M by.
#1 when gate open, 0 when gate closed.
#Instead of 0 or 1 use logistic regression to make it continuous and differentiable so
#we can use backprop.
#Call this a new Lowell machine.

#Regularizations which work on LSTM: L2, dropout (don't use on recurrent connections)
#What can you do: generate RNN prediction, sample from predictions, feed to next step.
#Predict, sample, predict, sample, ...
#You can sample multiple sequences at each step, and generate predictive distribution from
#each of those. Then you have multiple sequences "hypotheses", and pick most probable Hypotheses
#by computing probability of each character in each Hypothesis generated.
#Number of sequences grows exponentially, so only keep most likely few candidate sequences at each
#step -> Beam search.







