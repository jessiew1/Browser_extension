"""
Runs the kdd algorithm on the hard-toy.csv dataset.
Displays the current model on the training data.
Close the matplotlib window to run one additional iteration.
"""

from sklearn.model_selection import ShuffleSplit
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import random
import math

# reading toy data
df = pd.read_csv("../kdd-code/datasets/hard-toy.csv")
data = df.loc[:, ['x', 'y', 'bag', 'class']].values

# Flip classes so that positive is always at higher x values
# data[:,3] *= 2 * (data[:,2] >= 3) - 1

# Collect all data from the file
X = data[:,:2]
y = data[:,3]
bags = data[:,2]

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
    # y_train = y[train_index]
    y_test = y[test_index]
    R = bags[train_index]
    R_test = bags[test_index]

# log_reg = LogisticRegression(solver = 'lbfgs')

# def compute_theta(U, L_prime):
	# log_reg.fit(U, L_prime)
	# theta = None
	# return theta

# def compute_probs(U, theta):
	# probs = log_reg.predict_proba(U)[:, 1]
	# return probs

def add_bias(X):
	return np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)

# sign(x) = 1 if x >= 0 else 0
def sign(x):
	return 2 * (x >= 0) - 1

def compute_theta(U, L_prime):
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
		eta = 1e-4
		return w + eta * change
	
	theta = np.zeros((U.shape[1], 1))
	for _ in range(10):
		theta = train(U, L_prime, theta)

	return theta

def compute_probs(U, theta):
	U = add_bias(U)
	return np.matmul(U, theta)

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
	theta = compute_theta(U, L_prime)
	
	P = compute_probs(U, theta)
	L_prime = np.full(len(R), -1)
	t = np.full(len(B), 0.5)
	for r in all_bags:

		# Line 6
		bag = np.where(R == r)[0]
		if len(bag) == 0: continue
		ones = int(round(B[r] * len(bag)))
		bag_sorted = sorted(bag, key=lambda w: P[w], reverse=True)
		t[r] = P[bag_sorted[ones - 1]]

		# Line 7 and 8
		L_prime[bag_sorted[:ones]] = 1
		L_prime[bag_sorted[ones:]] = -1

	# The next line of code makes error is num mismatches of L_prime and L_prime_prime
	error = np.linalg.norm(L_prime - L_prime_prime, ord=0)
	# print('Sum and length of L\'', sum(L_prime), len(L_prime), error)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
	for r in all_bags:
		for c in [-1, 1]:
			plt.scatter([x for (x, y), r2, Lp in zip(U, R, L_prime) if r2 == r and Lp == c],[y for (x, y), r2, Lp in zip(U, R, L_prime) if r2 == r and Lp == c], marker = '$+$' if c == 1 else '$-$',c=colors[0])

		bias = theta[0] - t[r]
		weight_x = theta[1]
		weight_y = theta[2]
		xy = [(x, (-bias - x * weight_x) / weight_y) for x in range(math.floor(min(U[np.where(R == r)[0]][:,0])) - 1, math.ceil(max(U[np.where(R == r)[0]][:,0])) + 2)]
		xy = [(x, y) for x, y in xy if min(U[np.where(R == r)[0]][:,1]) - 1 <= y]
		xy = [(x, y) for x, y in xy if y <= max(U[np.where(R == r)[0]][:,1]) + 1]
		plt.plot([x for x, _ in xy], [y for _, y in xy], c=colors[0])
		
		colors = colors[1:]
	
	plt.xlim(min(U[:,0]) - 1, max(U[:,0]) + 1)
	plt.ylim(min(U[:,1]) - 1, max(U[:,1]) + 1)
	plt.title("KDD Algorithm intermediate model iteration {:d} on training data".format(it + 1))
	plt.show()
	
	plt.show()

	# Line 9
	if past_error is not None:
		if abs(past_error - error) < 0.05 * len(L_prime):
			break
	past_error = error


print('Model terminated in {:d} iterations'.format(it + 1))

new_y = np.full(U_test.shape[0], -1)
P = compute_probs(U_test, theta)
for r in all_bags:
	bag = np.where(R_test == r)[0]
	if len(bag) == 0: continue
	# print(r, bag, t[r], t)
	above_threshold_in_bag = [x for x in bag if P[x] >= t[r]]
	if len(above_threshold_in_bag) == 0: continue
	new_y[above_threshold_in_bag] = 1

print(classification_report(y_test, new_y))

# def logistic(x):
	# return 1 / (1 + np.exp(-x))

# U = train_data[:,:2]
# R = train_data[:,2]
# U = add_bias_per_bag(U, R)
# B = bag_proportions

# def train(X, t, w):
	# assert t.shape[1] == 1
	# assert t.shape[0] == X.shape[0]
	# assert w.shape[0] == X.shape[1]
	# assert w.shape[1] == 1

	# values = np.matmul(X, w)
	# predictions = sign(values)
	# misclassified = (predictions != t).astype(int)
	# change = np.matmul(np.transpose(X), misclassified * t)
	# eta = 1e-4
	# return w + eta * change

# # https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
# def train(X, t, w):
	# assert t.shape[1] == 1
	# assert t.shape[0] == X.shape[0]
	# assert w.shape[0] == X.shape[1]
	# assert w.shape[1] == 1

	# values = np.matmul(X, w)
	# eta = 1e-4
	# return w - eta * np.matmul(np.transpose(X), logistic(values) - (0.5 * t + 0.5))

# while True:
	# for _ in range(1000):
		# L_prime_prime = L_prime
		
		# theta = np.zeros((U.shape[1], 1))
		# for _ in range(100):
			# theta = train(U, L_prime.reshape((L_prime.shape[0], 1)), theta)
		
		# thresholds = dict()
		# for r in all_bags:
			# d = U[R == r]
			# Lpp = L_prime_prime[R == r]
			# values = monotone_P(d, theta).reshape(-1)
			# predictions = sign(values)
			# sorted_values = sorted(values)
			# percentile = 1 - B[r]
			# thresholds[r] = sorted_values[math.floor(percentile * len(sorted_values))]
		
		# L_prime = np.zeros(U.shape[0])
		# for u in range(U.shape[0]):
			# L_prime[u] = 1 if monotone_P(U[u:u+1,:], theta) >= thresholds[R[u]] else -1

	# print()
	# for r, bias in zip(all_bags, theta.reshape(-1)):
		# print('bias on {:d}: {:.4f}'.format(r, bias))
	# print('weight on x: {:.4f}'.format(theta[-2][0]))
	# print('weight on y: {:.4f}'.format(theta[-1][0]))
	
	# classes = set(map(lambda x: x[3], train_data))
	# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

	# for bag in all_bags:
		# for c2 in classes:
			# plt.scatter([x for (x, y, b, _), Lpu in zip(train_data, L_prime) if b == bag and Lpu == c2],[y for (x, y, b, _), Lpu in zip(train_data, L_prime) if b == bag and Lpu == c2], marker = '$1$' if c2 == 1 else '$0$',c=colors[0])
	
		
		# bias = theta[all_bags_inverse[bag]]
		# weight_x = theta[-2]
		# weight_y = theta[-1]
		# xy = [(x, (-bias - x * weight_x) / weight_y) for x in range(-10, 10)]
		# plt.plot([x for x, _ in xy], [y for _, y in xy], c=colors[0])
		
		# colors = colors[1:]
	
	# plt.xlim(-0.6, 4.2)
	# plt.ylim(-0.7, 2.6)
	
	# plt.show()
	
	# # If L_prime_prime is close to L_prime
	# matching = (L_prime == L_prime_prime)
	# print(np.sum(matching)/len(matching))
	# if False:
		# break
	
	
	

# print('The algorithm says to return theta!!!')
# print(theta)



















# train_X = data[:,:2]
# train_X = add_bias(train_X)

# train_t = data[:,3]
# train_t = np.reshape(train_t, (train_t.shape[0], 1))

# weights = np.zeros((train_X.shape[1], 1))
# for i in range(1 << 10):
	# weights = train(train_X, train_t, weights)
# print(weights)


