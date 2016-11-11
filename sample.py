import numpy as np 

class Sample:

	def __init__(self, predictors, target):
		""" Encapsulation of samples. 
		"""
		self.predictors = np.array(predictors)
		self.target = np.array(target)