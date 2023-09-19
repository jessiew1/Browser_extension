"""
Makes a synthetic dataset, runs the KDD algorithm on the dataset, and outputs the classification performance.
"""

from sklearn.model_selection import ShuffleSplit

import numpy as np
from collections import deque

import KDDAlgorithms
import LinAlg

# Data size
WEBSITES = 600
VOTING_REGIONS = 51
if VOTING_REGIONS >= 256: raise NotImplementedError
PEOPLE_PER_REGION = 2000
R = np.array([r for r in range(VOTING_REGIONS) for _ in range(PEOPLE_PER_REGION)]).astype(np.uint8)
TESTING_SPLIT = 0.2
BIAS_TYPE = 'nation'
assert BIAS_TYPE in ['none', 'nation', 'state']

# Ground truth
weights = np.random.normal(size = WEBSITES)
bias = 0		# Barring floating point precision error, the value of bias shouldn't matter because of the monotonicity of sigmoid in combination with the threshold mechanism of the KDD algorithm.

# Create voting region percentiles
B = np.random.normal(loc = 0.5, scale = 0.2, size = VOTING_REGIONS)
B = np.clip(B, 0.1, 0.9)
# B = np.random.uniform(size = VOTING_REGIONS)

# Create user data
U = np.random.dirichlet(np.ones((WEBSITES,)), size = (R.shape[0],))
assert (U.sum(1) > 1 - 1e-5).all()
assert (U.sum(1) < 1 + 1e-5).all()

# Normalize user data to unit length
U = U / np.sum(U, axis = 1).reshape((-1, 1))

# Score users by their data and weights and bias
scores = LinAlg.logistic(np.matmul(U, weights) + bias)

# Label users by applying percentiles of B to scores
L = np.zeros(R.shape[0])
for r in range(VOTING_REGIONS):
	bag = np.where(R == r)[0]
	if len(bag) == 0: raise RuntimeError('Voting region {:d} has no elements in the ground truth data.'.format(r))
	num_positive = int(round(B[r] * len(bag)))
	bag_sorted = sorted(bag, key=lambda w: scores[w], reverse=True)
	L[bag_sorted[:num_positive]] = 1
	L[bag_sorted[num_positive:]] = -1

# Split into training and testing data
rs = ShuffleSplit(n_splits=1, test_size=TESTING_SPLIT)
for train_index, test_index in rs.split(U):
	U_train = U[train_index,:]
	R_train = R[train_index]
	
	U_test = U[test_index,:]
	L_test = L[test_index]
	R_test = R[test_index]

import ClassifierImpls
classifier = ClassifierImpls.UnregularizedLogisticRegressionClassifier
compute_theta = classifier.compute_theta
compute_scores = classifier.compute_scores

def add_bias(U):
	return np.concatenate([np.ones((U.shape[0], 1)), U], axis = 1)

def add_bias_per_bag(U, R):
	out = deque()
	
	for u in range(len(U)):
		row = np.zeros(VOTING_REGIONS)
		row[R[u]] = 1
		row = np.concatenate([row, U[u]])
		out.append(row)

	out = np.stack(list(out))
	return out

# Add bias features to feature vectors
if BIAS_TYPE == 'state':
	compute_theta = (lambda f: lambda U, R, L_prime: f(add_bias_per_bag(U, R), R, L_prime))(compute_theta)
	compute_scores = (lambda f: lambda theta, U, R: f(theta, add_bias_per_bag(U, R), R))(compute_scores)
elif BIAS_TYPE == 'nation':
	compute_theta = (lambda f: lambda U, R, L_prime: f(add_bias(U), R, L_prime))(compute_theta)
	compute_scores = (lambda f: lambda theta, U, R: f(theta, add_bias(U), R))(compute_scores)

# Fit on training data
(theta, t) = KDDAlgorithms.algorithm_1(compute_theta, compute_scores, U_train, R_train, B)

# Measure performance on testing data
y_test = KDDAlgorithms.classify(compute_scores, (theta, t), U_test, R_test)
KDDAlgorithms.classification_report(L_test, y_test)