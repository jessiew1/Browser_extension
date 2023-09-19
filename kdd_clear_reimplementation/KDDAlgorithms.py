'''
Note: For the sake of code flexibility,
we deviate from the KDD paper and give the logistic
regression access to R, the voting region of each user.
This logistic regression implementation ignores the R parameter.
'''

import numpy as np

def algorithm_2(R, B):
	L_prime = np.full(len(R), -1)
	for r in range(len(B)):
		bag = np.where(R == r)[0]
		if len(bag) == 0: raise RuntimeError('Voting region {:d} has no elements in the training data.'.format(r))
		L_prime[bag] = 1 if B[r] >= 0.5 else -1
	return L_prime

def algorithm_1(compute_theta, compute_scores, U, R, B):
	# Stopping condition variable
	past_error = None
	
	# Line 1
	L_prime = algorithm_2(R, B)

	# Line 2
	for it in range(20):
	
		# Line 3
		L_prime_prime = np.copy(L_prime)
		
		# Line 4
		theta = compute_theta(U, R, L_prime)
		
		# Line 5 and 7
		scores = compute_scores(theta, U, R)
		t = np.full(len(B), 0.5)
		L_prime = np.full(len(R), -1)
		for r in range(len(B)):
		
			# Line 6
			bag = np.where(R == r)[0]
			if len(bag) == 0: raise RuntimeError('Voting region {:d} has no elements in the training data.'.format(r))
			num_positive = int(round(B[r] * len(bag)))
			bag_sorted = sorted(bag, key=lambda w: scores[w], reverse=True)
			t[r] = scores[bag_sorted[num_positive - 1]]
			
			# Line 7 and 8
			L_prime[bag_sorted[:num_positive]] = 1
			L_prime[bag_sorted[num_positive:]] = -1


		# Line 9 - stopping condition
		# error is the number of mismatches between L_prime and L_prime_prime
		error = int(np.linalg.norm(L_prime - L_prime_prime, ord=0))
		# print('Sum and length of L\' and error', sum(L_prime), len(L_prime), error)
		print(it, error)
		# if past_error is not None:
			# # print('past_error: {:d}'.format(past_error))
			# if past_error - error < 0.0001 * len(L_prime):
				# break
		past_error = error

	# Return Algorithm 1 parameters
	print('Model terminated in {:d} iterations'.format(it + 1))
	return (theta, t)

def classify(compute_scores, params, U, R):
	# Compute scores
	(theta, t) = params
	scores = compute_scores(theta, U, R)
	
	# Apply thresholds to scores
	labels = np.full(U.shape[0], -1)
	for r in range(int(max(R) + 1)):
		bag = np.where(R == r)[0]
		if len(bag) == 0: continue
		# print('Voting region {:d} has probability threshold {:.4f}'.format(r, t[r]))
		above_threshold = [x for x in bag if scores[x] - t[r] >= 0]
		if len(above_threshold) == 0: continue
		labels[above_threshold] = 1

	return labels

def classification_report(true_labels, predicted_labels):
	# Compute confusion matrix
	table = dict()
	for true_label, predicted_label in zip(true_labels, predicted_labels):
		if (predicted_label, true_label) not in table:
			table[(predicted_label, true_label)] = 0
		table[(predicted_label, true_label)] += 1
	
	# Fill empty confusion matrix cells with 0
	classes = set([a for b in table.keys() for a in b])
	for left in classes:
		for right in classes:
			if (left, right) not in table:
				table[(left, right)] = 0

	# Print header
	print()
	print('{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}'.format('', 'precision', 'recall', 'f1-score', 'support'))
	print()
	
	# Print per-class metrics
	for c in sorted(classes):
		precision = table[(c, c)] / sum([table[(c, a)] for a in classes])
		support = sum([table[(a, c)] for a in classes])
		recall = table[(c, c)] / support
		f1 = (precision + recall) / 2
		print('{:+11d}{:11.4f}{:11.4f}{:11.4f}{:11d}'.format(int(c), precision, recall, f1, support))
	
	# Print accuracy
	print()
	support = sum(table.values())
	accuracy = sum([table[(c, c)] for c in classes]) / support
	print('{:>11s}{:>11s}{:>11s}{:11.4f}{:11d}'.format('accuracy', '', '', accuracy, support))