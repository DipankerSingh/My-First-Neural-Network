======================================
## My First Neural Network 
======================================
In this project, I built a neural network from scratch to carry out a prediction problem 
on a real dataset! By building a neural network from the ground up, now I have a much 
better understanding of gradient descent, backpropagation and other concepts of NN.

The data comes from the UCI Machine Learning Database.

==============================================================================
***Problem Statement*** : Neural Network for predicting Bike Sharing Rides. Here NN will predict how many 
bikes a company needs because if they have too few they are losing money from potential riders and if they have
too many they are wasting money on bikes that are just sitting around. So NN will predict from the hisitorical 
data how many bikes they will need in the near future. 
==============================================================================

The network has two layers, a hidden layer and an output layer. The hidden layer uses the sigmoid function 
for activations. The output layer has only one node and is used for the regression, the output of the node is the 
same as the input of the node. That is, the activation function is  f(x)=xf(x)=x . A function that takes the input 
signal and generates an output signal, but takes into account the threshold, is called an activation function. We 
work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer 
become inputs to the neurons on the next layer. This process is called ***forward propagation.*** We use the weights to 
propagate signals forward from the input to the output layers in a neural network. We use the weights to also 
propagate error backwards from the output back into the network to update our weights. This is called ***backpropagation.***

================================================================================
#Hyperparameters:

1) epochs = 5000
2) learning_rate = 0.01
3) hidden_nodes = 28
4) output_nodes = 1

================================================================================
#Performance:

 
===============================================================================

#Take aways :

1) Why Bias Term is Useful : 
At a very high level, it is the difference between y = mx and y = mx + c (where c is the bias term). The first equation 
can only describe data where (0, 0) is part of the dataset, while the second equation has no such restriction. Typically 
in a neural network, the second equation becomes of the form y = w1x + w0b (where b is the bias term, which can be 
turned on and off by appropriate weight on w0)
Read more : 
** http://stats.stackexchange.com/questions/185911/why-are-bias-nodes-used-in-neural-networks **

2) Sigmoid functions add the non-linearity in our network. Why do we need non-linearity anyway?
In one line : "no matter how many layers would behave just like a single perceptron (because linear functions 
added together just give you a linear function)" 
Read more :
**http://stackoverflow.com/a/9783865**

3) Great Explanation about Backpropogation:
Here is a great explanation about the backpropagation algorithm that I found helpful to get some intuition.
Please read it at your leisure and don't worry if you don't understand it completely. As the author says you 
get finer appreciation for the algorithm over many days and weeks :)

http://neuralnetworksanddeeplearning.com/chap2.html

4) Hyperparameters: 
4a) The number of epochs is chosen such the network is trained well enough to accurately make predictions but is not 
overfitting to the training data.

Image of Result:

I found out that I don't really need to perform 5000 epochs; I could have stopped earlier

4b) The number of hidden units is chosen such that the network is able to accurately predict the number of bike riders, 
is able to generalize, and is not overfitting.

The choice of 28 seem like a reasonable choice. Here is the general guideline:

The number of nodes is fairly open; if validation loss is low, it should meet specifications. It should probably be no more than twice the number of input units, 
and enough that the network can generalize, so probably at least 8. A good rule of thumb is the half way in between the number of input and output units.

There's a good answer here for how to decide the number of nodes in the hidden layer.
https://www.quora.com/How-do-I-decide-the-number-of-nodes-in-a-hidden-layer-of-a-neural-network

4c) The learning rate is chosen such that the network successfully converges, but is still time efficient.

A small learning rate makes the network take a longer time to train. A large learning rate may make the network not converge.

Effect of learning rate on Gradient descent:
http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/

Finding a good learning rate is more of an art than science. Choice of 0.01 for learning rate seems reasonable.

===============================================================================
#Results:

The final predictions were good for most of the part except for the period of Dec-21 to Dec-27

Result Image

So the possible reason could be : Our training dataset has less examples for holidays, and that's why our model could not predict well for these days.

This is the correct intuition. It can be solidified more by looking at this graph:

Image

The 'cnt' variable shows two different regimes (one for the rest of the year) and (one for the last ten days of the year.) Notice also that we took away the
 last 80 days of the data as validation and test data. Thus the model has only one dec period to learn from. Is that enough data to learn that the demand 
drops off in December? However it has a lot of monthly data that is pretty much cyclical, so it can figure out that demand is high on weekdays or on a Monday etc. 
since it has enough of that data. It just doesn't have enough data to figure out the 'annual' patterns.

===============================================================================

***One More Question : what is the tradeoff between the no. of layers and no of neurons in each layer ? What should we prefer ?***
Answer given by my reviewer : 
Also regarding the tradeoff, I am not sure about the answer, but I think its false dichotomy. I wasn't aware if someone 
was rationing out the hidden nodes :wink: Why couldn't you use both, A large number of hidden nodes and a large number of layers?
 In general Neural Networks are very powerful and it would be very uncommon to see a Neural Network with multiple layers for a
 simple problem. Further because of the properties of the sigmoid the weight updates (and the errors) in a Neural network diminish
 as they are back-propagated though multiple layers. This is called as vanishing gradient problem 
(https://en.wikipedia.org/wiki/Vanishing_gradient_problem) and hence a network with more layers usually is harder to train than
 one with just one layer but a lot of Neurons. But a Neural Network with many layers learns different attributes of the data in each 
layer (and hence ends up being more powerful) I hope that added some color to the problem for you?

==============================================================================
1) To Create a new conda environment:

conda create --name dipsNN python=3
============================

2) Enter the environment:

Mac/Linux: >> source activate dipsNN
Windows: >> activate dipsNN
===========================

3) Ensure you have numpy, matplotlib, pandas, and jupyter notebook installed by doing the following:

conda install numpy matplotlib pandas jupyter notebook
=========================================

4) Run the following to open up the notebook:

jupyter notebook dipsNN-my-first-neural-network.ipynb
=========================================

====================================================================
##Special Thanks to the reviewer of the Udacity Team for his guidance and all his resourcefull, and in depth reviews
====================================================================

