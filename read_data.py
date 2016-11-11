"""
read_data.py contains functions to read in the text data, and turn it into
useable forms for neuralnetwork.py.
"""

from sample import Sample


def read_data(file_location):
	""" Reads data and returns a list of the results. 
	"""
	with open(file_location) as f:
		data = f.read().splitlines()
	data = [line.split(",") for line in data]
	return data


def hot_encode(data):
	""" Converts all data into integers, and hot encodes the output into the form
	of a ten element array. The elements of the response array are all 0, with the
	exception of the actual output, which shows up as a 1.
	"""
	init_len_sample = len(data[0])

	for i in range(len(data)):
		for k in range(init_len_sample-1):
			data[i][k] = int(data[i][k])/16
		data[i][-1] = int(data[i][-1])

	for i in range(len(data)):
		sample = []
		sample.append(data[i][:-1])
		label = data[i][-1]
		hot_encoded = [0]*10
		hot_encoded[label] = 1
		sample.append(hot_encoded)
		data[i] = sample


def to_object(data):
	""" Converts hot coded array into an array of Samples, an encapsulation of
	training/testing data. 
	"""
	sample_arr = []
	for i in range(len(data)):
		sample_arr.append( Sample( data[i][0], data[i][1] ) )
	return sample_arr