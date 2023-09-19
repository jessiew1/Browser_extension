"""
Runs the kdd algorithm on the hard-toy.csv dataset.
Displays the final trained model on the test data.
"""

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import random
import math

BIAS_TYPE = 'nation'
assert BIAS_TYPE in ['none', 'nation', 'state']

'''
bias  ,  prN,  prP,  rcN,  rcP, suppN, suppP
none  : 0.61, 0.54, 0.56, 0.59,   128,   112
nation: 0.61, 0.55, 0.58, 0.58,   128,   112
state : 0.61, 0.54, 0.56, 0.59,   128,   112
'''



# reading toy data
df = pd.read_csv("../kdd-code/datasets/hard-toy.csv")
data = df.loc[:, ['x', 'y', 'bag', 'class']].values

# Flip classes so that positive is always at higher x values
# data[:,3] *= 2 * (data[:,2] >= 3) - 1

# Collect all data from the file
X = data[:,:2]
y = data[:,3]
bags = data[:,2].astype(np.uint8)

# If the assertion holds, then you can use all_bags as both array indices and symbolic values
all_bags = sorted(set(map(lambda x: x[2], data)))
assert all_bags == list(range(len(all_bags)))
all_bags = list(map(int, all_bags))

# Compute proportions
B = dict()
for r in all_bags:
	bag = np.where(bags == r)[0]
	B[r] = np.sum(y[bag] == 1) / len(bag)

# Split into training and testing data
test_size = 0.2
rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
for train_index, test_index in rs.split(X):
    U = X[train_index, :]
    U_test = X[test_index, :]
    y_train = y[train_index]
    y_test = y[test_index]
    R = bags[train_index]
    R_test = bags[test_index]

def add_bias(X):
	return np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)

def add_bias_per_bag(X, R):
	out = []
	
	for u in range(len(X)):
		row = np.zeros(len(all_bags))
		row[R[u]] = 1
		row = np.concatenate([row, X[u]])
		out += [row]
	out = np.stack(out)
	return out

# sign(x) = 1 if x >= 0 else 0
def sign(x):
	return 2 * (x >= 0) - 1

def logistic(x):
	return 1/(1 + np.exp(-x))

def inverse_logistic(x):
	return np.log(x / (1 - x))

# Perceptron algorithm, Bishop 4.1.7
def compute_theta(U, R, L_prime):
	if BIAS_TYPE == 'state':
		U = add_bias_per_bag(U, R)
	elif BIAS_TYPE == 'nation':
		U = add_bias(U)
	
	def train(X, t, w):
		t = t.reshape((-1, 1))
		assert t.shape[1] == 1
		assert t.shape[0] == X.shape[0]
		assert w.shape[0] == X.shape[1]
		assert w.shape[1] == 1

		values = np.matmul(X, w)
		predictions = sign(values)
		misclassified = (predictions != t).astype(int)
		change = np.matmul(np.transpose(X), misclassified * t)
		eta = 1e-3
		return w + eta * change
	
	theta = np.zeros((U.shape[1], 1))
	for _ in range(10):
		theta = train(U, L_prime, theta)

	return theta

# Logistic regression, sigmoid and cross entropy loss
# def compute_theta(U, R, L_prime):
	# if BIAS_TYPE == 'state':
		# U = add_bias_per_bag(U, R)
	# elif BIAS_TYPE == 'nation':
		# U = add_bias(U)
		
	# # https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
	# # https://web.stanford.edu/~jurafsky/slp3/5.pdf (Eq. 5.17)
	# def train(X, t, w):
		# t = t.reshape((-1, 1))
		# assert t.shape[1] == 1
		# assert t.shape[0] == X.shape[0]
		# assert w.shape[0] == X.shape[1]
		# assert w.shape[1] == 1

		# values = np.matmul(X, w)
		# gradient = np.matmul(np.transpose(X), logistic(values) - (0.5 * t + 0.5))
		# eta = 1e-2
		# return w - eta * gradient

	# theta = np.zeros((U.shape[1], 1))
	# for _ in range(10):
		# theta = train(U, L_prime, theta)

	# return theta

def compute_probs(U, R, theta):
	if BIAS_TYPE == 'state':
		U = add_bias_per_bag(U, R)
	elif BIAS_TYPE == 'nation':
		U = add_bias(U)
	values = np.matmul(U, theta)
	return logistic(values)

# Line 1
L_prime = np.full(len(R), -1)
for r in all_bags:
	bag = np.where(R == r)[0]
	if len(bag) == 0: continue
	L_prime[bag] = 1 if B[r] >= 0.5 else -1

# Line 2
past_error = None
for it in range(100):
	# Line 3
	L_prime_prime = L_prime.copy()

	# Line 4
	theta = compute_theta(U, R, L_prime)
	
	P = compute_probs(U, R, theta)
	L_prime = np.full(len(R), -1)
	t = np.full(len(B), 0.5)
	for r in all_bags:

		# Line 6
		bag = np.where(R == r)[0]
		if len(bag) == 0: continue
		num_positive = int(round(B[r] * len(bag)))
		bag_sorted = sorted(bag, key=lambda w: P[w], reverse=True)
		t[r] = P[bag_sorted[num_positive - 1]]

		# Line 7 and 8
		L_prime[bag_sorted[:num_positive]] = 1
		L_prime[bag_sorted[num_positive:]] = -1

	# Line 9
	# The next line of code makes error the number of points whose putative labels were changed
	error = np.linalg.norm(L_prime - L_prime_prime, ord=0)
	if past_error is not None:
		if abs(past_error - error) < 0.05 * len(L_prime):
			break
	past_error = error

print('Model terminated in {:d} iterations'.format(it))

# Display decision boundaries
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for r in all_bags:
	for c in [-1, 1]:
		plt.scatter([x for (x, y), r2, label in zip(U_test, R_test, y_test) if r2 == r and label == c],[y for (x, y), r2, label in zip(U_test, R_test, y_test) if r2 == r and label == c], marker = '$+$' if c == 1 else '$-$',c=colors[0])

	if BIAS_TYPE == 'state':
		bias = theta[r]-inverse_logistic(t[r])
	elif BIAS_TYPE == 'nation':
		bias = theta[0]-inverse_logistic(t[r])
	else:
		bias = -inverse_logistic(t[r])
	weight_x = theta[-2]
	weight_y = theta[-1]
	xy = [(x, (-bias - x * weight_x) / weight_y) for x in range(math.floor(min(U[np.where(R == r)[0]][:,0])) - 1, math.ceil(max(U[np.where(R == r)[0]][:,0])) + 2)]
	xy = [(x, y) for x, y in xy if min(U[np.where(R == r)[0]][:,1]) - 1 <= y]
	xy = [(x, y) for x, y in xy if y <= max(U[np.where(R == r)[0]][:,1]) + 1]
	plt.plot([x for x, _ in xy], [y for _, y in xy], c=colors[0])
	colors = colors[1:]
plt.xlim(min(U[:,0]) - 1, max(U[:,0]) + 1)
plt.ylim(min(U[:,1]) - 1, max(U[:,1]) + 1)
plt.title("KDD Algorithm final model on testing data")
plt.show()

# Measure performance on testing data
new_y = np.full(U_test.shape[0], -1)
P = compute_probs(U_test, R_test, theta)
for r in all_bags:
	bag = np.where(R_test == r)[0]
	if len(bag) == 0: continue
	# print(r, bag, t[r], t)
	above_threshold_in_bag = [x for x in bag if P[x] - t[r] >= 0]
	if len(above_threshold_in_bag) == 0: continue
	new_y[above_threshold_in_bag] = 1
print(classification_report(y_test, new_y))