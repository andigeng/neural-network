# neural-network
A Python implementation of a fully-connected feedforward neural network. Currently uses sigmoidal activation functions for each node, and only implements per-pattern learning for classification (although it'd be a simple modification for non-classification tasks).

### To intialize a neural network:
```python
layers = [10,20,10]
learning_rate = 0.25
min_initial_weight = -0.3
max_initial_weight = 0.3

net = NeuralNet(layers, learning_rate, min, min_initial_weight, max_initial_weight)
```
Layers is an array that holds the number of nodes in each layer, with the first and last numbers representing the number of inputs and outputs, respectively.

### To train a network:
```python
num_epochs = 10
net.train_network(data, num_epochs)
```

### Running the demo:
Clone the repo, navigate to the directory and run demo.py. 
```
python demo.py
```
This will train the network on a preprocessed version of the MNIST dataset.
