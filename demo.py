from neuralnet import NeuralNet
import read_data as rd
from sample import Sample

def main():
	
	# Imports and converts training and test data to useable form
	training_data = rd.read_data('data/training.txt')
	rd.hot_encode(training_data)
	training_data = rd.to_object(training_data)

	test_data = rd.read_data('data/testing.txt')
	rd.hot_encode(test_data)
	test_data = rd.to_object(test_data)

	# Initialize neural network
	net = NeuralNet([64, 90, 10], 0.25, -0.3, 0.3)

	# Train neural network with 5 epochs
	net.train_network(training_data, 5)

	# Display accuracies for training and testing dataset
	print('\nFinal Testing Accuracy')
	print(net.accuracy(test_data))

	print('\nFinal Training Accuracy:')
	print(net.accuracy(training_data))


if __name__ == "__main__":
	main()