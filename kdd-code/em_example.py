'''
This code uses the original paper's implementation and displays the results.

Each bag has its own color.
Labels are + or -.
The line displays the learned decision boundary for each bag.
'''


from llp.kdd import EM

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
from llp.util import compute_proportions

import matplotlib.pyplot as plt

fit_intercept = False

# reading toy data
df = pd.read_csv("datasets/hard-toy.csv")
X = df.loc[:, ["x", "y"]].values
y = df.loc[:, ["class"]].values.reshape(-1)
bags = df.loc[:, ["bag"]].values.reshape(-1)
proportions = compute_proportions(bags, y)

# If the assertion holds, then you can use all_bags as both array indices and symbolic values
all_bags = sorted(set(bags))
assert all_bags == list(range(len(all_bags)))
all_bags = list(map(int, all_bags))

# To train the model, you need, X, bags and proportions
# To test the model, you need, X and bags
# If you have ground-truth, y, you can evaluate your predictions

# train/test split
# test_size = 1/3
test_size = 0.2
rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
for train_index, test_index in rs.split(X):
	X_train = X[train_index, :]
	X_test = X[test_index, :]
	y_train = y[train_index]
	y_test = y[test_index]
	bags_train = bags[train_index]
	bags_test = bags[test_index]

	# Using a SVM instead of Logistic regression
	# model = EM(SVC(gamma = 'scale', C = 1))
	# model.fit(X_train, bags_train, proportions)
   
	model = EM(LogisticRegression(solver = 'lbfgs', fit_intercept = fit_intercept))
	model.fit(X_train, bags_train, proportions)

	new_y = model.predict(X_test, bags_test)
	print(classification_report(y_test, new_y))

	import math
	
	U = X_train
	R = bags_train
	if fit_intercept:
		theta = np.concatenate([model.model.intercept_, model.model.coef_[0]])
	else:
		theta = model.model.coef_[0]
	L_prime = model.predict(U, R)

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
	for r in all_bags:
		for c in [-1, 1]:
			plt.scatter([x for (x, y), r2, Lp in zip(U, R, L_prime) if r2 == r and Lp == c],[y for (x, y), r2, Lp in zip(U, R, L_prime) if r2 == r and Lp == c], marker = '$+$' if c == 1 else '$-$',c=colors[0])

		def logit(x):
			return math.log(x / (1 - x))

		if fit_intercept:
			bias = theta[0] - logit(model.t[r])
		else:
			bias = -logit(model.t[r])
		weight_x = theta[-2]
		weight_y = theta[-1]
		xy = [(x, (-bias - x * weight_x) / weight_y) for x in range(math.floor(min(U[np.where(R == r)[0]][:,0])) - 1, math.ceil(max(U[np.where(R == r)[0]][:,0])) + 2)]
		xy = [(x, y) for x, y in xy if min(U[np.where(R == r)[0]][:,1]) - 1 <= y]
		xy = [(x, y) for x, y in xy if y <= max(U[np.where(R == r)[0]][:,1]) + 1]
		plt.plot([x for x, _ in xy], [y for _, y in xy], c=colors[0])
		colors = colors[1:]
	
	plt.xlim(min(U[:,0]) - 1, max(U[:,0]) + 1)
	plt.ylim(min(U[:,1]) - 1, max(U[:,1]) + 1)
	
	plt.show()
