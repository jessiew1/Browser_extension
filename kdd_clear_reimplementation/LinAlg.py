'''
Linear algebra functions for numpy arrays.
'''

import numpy as np

# sign(x) = 1 if x >= 0 else 0
def sign(x):
	return 2 * (x >= 0) - 1

# The sigmoid function. logistic is the approximation of this.
# This is used only for the ground truth.
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# The squashing function that other files in this folder will use.
def logistic(x):
	# Check for potential floating point numerical instability
	if np.max(np.abs(x)) > 20:
		print('logistic applied to {:+.3f} to {:+.3f}'.format(np.min(x), np.max(x)))
	
	# Use the sigmoid function.
	return sigmoid(x)
