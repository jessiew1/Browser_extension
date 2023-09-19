"""
Runs the kdd algorithm on the hard-toy.csv dataset.
After each iteration of the KDD algorithm, it prints debugging information.
Close the matplotlib window to run one additional iteration.
"""

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

indices = []
while len(indices) < 0.8 * len(data):
	index = random.randint(0, len(data) - 1)
	if index in indices: continue
	indices += [index]

train_data = np.stack([data[i] for i in indices])
test_data = np.stack([data[i] for i in range(len(data)) if i not in indices])

# Collect set of bags
all_bags = sorted(set(map(lambda x: x[2], data)))
all_bags = list(map(int, all_bags))
all_bags_inverse = {v: k for k, v in enumerate(all_bags)}

# Compute bag proportions
bag_proportions = dict()
for bag in all_bags:
	d = train_data[train_data[:,2] == bag]
	positive = len(d[d[:,3] == 1])
	total = len(d)
	assert total > 0
	bag_proportions[bag] = positive / total

def logistic(x):
	return 1 / (1 + np.exp(-x))

def add_bias(X):
	return np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)

def add_bias_per_bag(X, R):
	out = []
	
	for u in range(len(X)):
		row = np.zeros(len(all_bags))
		row[all_bags_inverse[R[u]]] = 1
		row = np.concatenate([row, X[u]])
		out += [row]
	out = np.stack(out)
	return out

U = train_data[:,:2]
R = train_data[:,2]
U = add_bias_per_bag(U, R)
B = bag_proportions

def algorithm_2(U, R, B):
	L_prime = np.zeros(U.shape[0])

	for u in range(R.shape[0]):
		# My perceptron weight update says to update by zero if I did the real Algorithm 2
		L_prime[u] = 1 if B[R[u]] >= 0.5 else -1

	return L_prime

L_prime = algorithm_2(U, R, B)

# sign(x) = 1 if x >= 0 else 0
def sign(x):
	return 2 * (x >= 0) - 1

def train(X, t, w):
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

# https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
def train(X, t, w):
	assert t.shape[1] == 1
	assert t.shape[0] == X.shape[0]
	assert w.shape[0] == X.shape[1]
	assert w.shape[1] == 1

	values = np.matmul(X, w)
	eta = 1e-4
	return w - eta * np.matmul(np.transpose(X), logistic(values) - (0.5 * t + 0.5))

def monotone_P(X, w):
	return np.matmul(X, w)

while True:
	for _ in range(1):
		L_prime_prime = L_prime
		
		theta = np.zeros((U.shape[1], 1))
		for _ in range(10):
			theta = train(U, L_prime.reshape((L_prime.shape[0], 1)), theta)
		
		thresholds = dict()
		for r in all_bags:
			d = U[R == r]
			Lpp = L_prime_prime[R == r]
			values = monotone_P(d, theta).reshape(-1)
			predictions = sign(values)
			sorted_values = sorted(values)
			percentile = 1 - B[r]
			thresholds[r] = sorted_values[math.floor(percentile * len(sorted_values))]
		
		L_prime = np.zeros(U.shape[0])
		for u in range(U.shape[0]):
			L_prime[u] = 1 if monotone_P(U[u:u+1,:], theta) >= thresholds[R[u]] else -1

	print()
	for r, bias in zip(all_bags, theta.reshape(-1)):
		print('bias on {:d}: {:.4f}'.format(r, bias))
	print('weight on x: {:.4f}'.format(theta[-2][0]))
	print('weight on y: {:.4f}'.format(theta[-1][0]))
	
	classes = set(map(lambda x: x[3], train_data))
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

	for bag in all_bags:
		for c2 in classes:
			plt.scatter([x for (x, y, b, _), Lpu in zip(train_data, L_prime) if b == bag and Lpu == c2],[y for (x, y, b, _), Lpu in zip(train_data, L_prime) if b == bag and Lpu == c2], marker = '$1$' if c2 == 1 else '$0$',c=colors[0])
	
		
		bias = theta[all_bags_inverse[bag]] - thresholds[all_bags_inverse[bag]]
		weight_x = theta[-2]
		weight_y = theta[-1]
		xy = [(x, (-bias - x * weight_x) / weight_y) for x in range(-10, 10)]
		plt.plot([x for x, _ in xy], [y for _, y in xy], c=colors[0])
		
		colors = colors[1:]
	
	plt.xlim(min(train_data[:,0]) - 1, max(train_data[:,0]) + 1)
	plt.ylim(min(train_data[:,1]) - 1, max(train_data[:,1]) + 1)
	
	plt.show()
	
	# If L_prime_prime is close to L_prime
	matching = (L_prime == L_prime_prime)
	print(np.sum(matching)/len(matching))
	if False:
		break
	
	
	

print('The algorithm says to return theta!!!')
print(theta)



















# train_X = data[:,:2]
# train_X = add_bias(train_X)

# train_t = data[:,3]
# train_t = np.reshape(train_t, (train_t.shape[0], 1))

# weights = np.zeros((train_X.shape[1], 1))
# for i in range(1 << 10):
	# weights = train(train_X, train_t, weights)
# print(weights)


