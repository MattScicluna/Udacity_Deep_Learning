#  Udacity Deep Learning Course Notes


#  Lecture 3: Convolutional Neural Networks


#######################################
####  3/13 Statistical Invariance  ####
#######################################

#Solved by weight sharing. For images: Convolutional Neural Networks (Convnets)
#For text/sequences: Embeddings and Recurrent Neural Networks

##############################################
####  4/13 Convolutional Neural Networks  ####
##############################################

#Images are Dimension: Height x Width x Depth
#Depth is usually 3 = RBG colour channels.

#Run same Neural network on m x n patch of image (sometimes called Kernels), with k outputs 
#(each of the k output is a feature map, and k is a measure of Semantic Complexity),
#now you have image with Height = m, Width = n, and Depth = k.
#Apply convolutions progressively - looks like a pyramid 
#(smaller Height and Width but deeper Depth.)

#Stride = #pixels you're shifting each time you move your filter.
#Stride = 1 -> output same size as input.
#Stride = 2 -> output about half size as input.

#Valid Padding -> patch doesn't go off edge
#Same Padding -> patch goes off edge, and pads missing values with 0's 
#	      -> output map size = input map size

##################################
####  5/13 Feature Map Sizes  ####
##################################

#Given Input Dimensions = 28 x 28 x 3
#      Output Dimensions = 3 x 3 x 8

#|Padding | Stride | Width | Height | Depth |
#|'Same'  | 1      | 28    | 28     | 8     |
#|'Valid' | 1      | 26    | 26     | 8     |
#|'Valid' | 2      | 13    | 13     | 8     |

#######################################
####  6/13 Convolutions Continued  ####
#######################################

#Chain rule with sharing:
#Add the gradients for every patch to give them all the same value

###########################################
####  7/13 Exploring the Design Space  ####
###########################################

#Pooling->Alternative to using large strides
#       ->Take all convolutions in a neighbourhood and combine them
#       ->Max -> Parameter free
#             -> More expensive (since small strides)
#             -> more hyperparameters (pooling size + stride)
#       ->Average -> "Blurred" low resolution view of image

###################################
####  8/13 1 x 1 Convolutions  ####
###################################

#For n x m image, n x m convolution is same as regular neural network
#1 x 1 Convolutions -> do a m x n convolution and run a 1 x 1 convolution on the k outputs. This is a deep network over the patch!
#Cheap way of making model deeper since 1 x 1 convolutions are just Matrix Multiplications.

#################################
####  9/13 Inception Module  ####
#################################

#Inception Modules
#At each layer of your Convnet, do Pooling and Convolutions with different sizes, then concatenate all of the features at the top layer.




