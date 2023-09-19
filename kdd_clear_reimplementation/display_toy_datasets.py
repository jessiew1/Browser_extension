'''
This file displays the data from datasets/hard-toy-bivariate.csv.
Data points are real-valued (x, y) pairs and labels are 0 or 1.
Each bag is shown, one by one, and then all bags are shown at once.
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

FILE_TO_READ = "../kdd-code/datasets/hard-toy-bivariate.csv"

# reading toy data
df = pd.read_csv(FILE_TO_READ)
data = df.loc[:, ["x", "y", 'bag', 'class']].values

# # Flip classes so that positive is always at higher x values
# data[:,3] *= 2 * (data[:,2] >= 3) - 1

print(data)

# Collect the set of all bag values and all class values
bags = set(map(lambda x: x[2], data))
classes = set(map(lambda x: x[3], data))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for bag in bags:
	for c2 in classes:
		plt.scatter(
			[x for x, y, b, c in data if b == bag and c == c2],
			[y for x, y, b, c in data if b == bag and c == c2],
			marker = '$1$' if c2 == 1 else '$0$',
			c=colors[0],
		)
	colors = colors[1:]
	plt.title("hard-toy-bivariate - Bag {:d}".format(int(round(bag))))
	plt.show()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for bag in bags:
	for c2 in classes:
		plt.scatter(
			[x for x, y, b, c in data if b == bag and c == c2],
			[y for x, y, b, c in data if b == bag and c == c2],
			marker = '$1$' if c2 == 1 else '$0$',
			c=colors[0],
		)
	colors = colors[1:]
plt.title("hard-toy-bivariate - All Data")
plt.show()
