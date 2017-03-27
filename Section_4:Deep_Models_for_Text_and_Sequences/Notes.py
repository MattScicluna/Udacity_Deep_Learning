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
