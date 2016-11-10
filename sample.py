import numpy as np 

class Sample:

	def __init__( self, predictors, target ):
		""" """
		self.predictors = np.array( predictors )
		self.target = np.array( target )