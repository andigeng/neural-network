"""
neuralnet.py contains an implementation of a fully connected feedforward neural
network. The network only implements sigmoid activation function, and must be 
fullyconnected. Only per-pattern training is implemented as well.
"""

import numpy as np
from sample import Sample



class NeuralNet:
  """ Class to store and create a fully connected feedforward neural network.""" 

  def __init__(self, neuron_arr, learning_rate=None, weight_min=None, 
                weight_max=None):
    """ Initializes the fully connected feedforward network. 
    """

    # Initalize matrices that will keep track of neural network's data.
    self.weights = []
    self.inputs = []
    self.outputs = []
    self.errors = []

    if learning_rate: self.learning_rate = learning_rate
    else: self.learning_rate = 0.25
    if (weight_min == None): weight_min = -0.3
    if (weight_max == None): weight_max = 0.3

    self.neuron_arr = neuron_arr
    self.num_layers = len(self.neuron_arr)

    num_weight_layers = len(neuron_arr) - 1

    # Creates a matrix that contains randomly initialized weights.
    for layer in range(num_weight_layers):
      self.weights.append(
        np.random.uniform(
            low   = weight_min,
            high  = weight_max,
            size  = (neuron_arr[layer + 1], neuron_arr[layer] + 1)
          )
        )
    # Creates the input, output, and error matrices.
    self.create_neuralnet_arrays()


  def create_neuralnet_arrays(self):
    """ Specifies the dimensions of the matrices that represent the values and
    weights of the neural network.
    """

    for layer in range(self.num_layers):
      self.inputs.append(np.empty([1, self.neuron_arr[layer]]))
      self.errors.append(np.empty([1, self.neuron_arr[layer]]))

      # Add a bias node (always on) if we are not on the last layer.
      if (layer < self.num_layers - 1):
        self.outputs.append(np.empty([1, self.neuron_arr[layer] + 1]))
      else:
        self.outputs.append(np.empty([1, self.neuron_arr[layer]]))


  def train_network(self, dataset, num_epoch):
    """ Implements per-pattern learning. Runs through training dataset a
    specified number of times, and prints training accuracy after each
    epoch. The dataset is also scrambled after each epoch.
    """

    print("Untrained -- Training Set Accuracy: {}".format(
        self.accuracy(dataset)))
    
    for epoch in range(num_epoch):
      np.random.shuffle(dataset)
      
      for sample in dataset:
        self.forward_pass(sample.predictors)
        self.backprop(sample.target)
      
      print("Epoch {} -- Training Set Accuracy: {}".format(
          epoch+1,
          self.accuracy(dataset)
        ))


  def forward_pass(self, inputs):
    """ Perform a forward pass through the neural network. Input and output
    matrices are updated. Note that operations are vectorized using numpy.
    """

    # Adds a vector composed of ones onto input -- acts as bias node
    self.outputs[0] = np.concatenate((np.ones(1), inputs))

    for layer in range(1, self.num_layers):
      # Calculates each successive layers' inputs -- the dot product of all the
      # incoming values and weights, which then go through sigmoid function
      transposed_weights = np.transpose(self.weights[layer - 1])
      self.inputs[layer] = np.dot(self.outputs[layer - 1], transposed_weights)
      temp_output = self.sigmoid_activation(self.inputs[layer])

      # If we are not on the last layer, include bias node
      if (layer < self.num_layers - 1):
        self.outputs[layer] = np.concatenate((np.ones(1), temp_output))
      else:
        self.outputs[layer] = temp_output
  

  def backprop(self, outputs):
    """ Performs one pass backwards through the network. Weight and error
    matrices are updated. Note that operations are vectorized using numpy.
    """

    # Updates the error matrix with a vector of the norm errors
    self.errors[-1] = self.outputs[-1] - outputs

    for i in range( self.num_layers - 2, 0, -1):
      # Error values flow backwards proportional to the weight of the connections
      # which are also proportional to the original input values
      forward_pass_input = self.sigmoid_deriv(self.inputs[i])
      error_dot_connections = np.dot(self.errors[i + 1], self.weights[i][:, 1:])
      self.errors[i]  = forward_pass_input * error_dot_connections
    
    for i in range(0, self.num_layers - 1):
      # The gradient is the dot product of the errors flowing backwards with the
      # values of the node it is connected to
      error_vector = np.array([self.errors[i + 1]])
      output_vector = np.array([self.outputs[i]])
      transposed_errors = np.transpose(error_vector)

      gradient = np.dot(transposed_errors, output_vector)
      self.weights[i] -= self.learning_rate * gradient


  def sigmoid_activation(self, num):
    """ Computes output of sigmoid function given a number. 
    """
    result = 1/(1 + np.exp(-num))
    return result


  def sigmoid_deriv(self, num):
    """ Computes the derivative of the sigmoid function given a number. 
    """
    result = self.sigmoid_activation(num) * (1 - self.sigmoid_activation(num))
    return result


  def make_pred(self, input):
    """ Given an input, returns the predicted value of the network. 
    """
    self.forward_pass(input)
    output_arr = self.outputs[-1]
    return np.argmax(output_arr)


  def accuracy(self, dataset):
    """ Iterates through the given dataset, and returns the accuracy of the 
    network. 
    """
    num_correct = 0

    for sample in dataset:
      prediction = self.make_pred(sample.predictors)

      if (sample.target[prediction] == 1):
        num_correct += 1
    
    return (num_correct/len(dataset))