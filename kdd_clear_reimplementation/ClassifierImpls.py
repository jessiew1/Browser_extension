'''
Implemented classifiers:
SKLearnClassifier
UnregularizedLogisticRegressionClassifier
	https://web.stanford.edu/~jurafsky/slp3/5.pdf (Eq. 5.17)
	L2 Regularization is commented out.
	Constants are hardcoded
PerceptronClassifier
	Bishop 4.1.7
	Constants are hardcoded

These are tuples of size 2.
First element is compute_theta.
Second element is compute_scores.
'''

import numpy as np
import LinAlg

from sklearn.linear_model import LogisticRegression

class Classifier:
	def __init__():
		raise RuntimeError("Call the methods directly. Do not initialize an object.")
		
	def compute_theta(U, R, L_prime):
		raise NotImplementedError
		
	def compute_scores(theta, U, R):
		raise NotImplementedError

class SKLearnClassifier(Classifier):
	"""
	Transferring theta from compute_theta to compute_scores has not been tested.
	"""
	def compute_theta(U, R, L_prime):
		model = LogisticRegression(solver = 'lbfgs', max_iter = 1 << 14)
		model.fit(U, L_prime)
		theta = model.get_params()
		return theta
		
	def compute_scores(theta, U, R):
		model = LogisticRegression(solver = 'lbfgs', max_iter = 1 << 14)
		model.set_params(theta)
		return model.predict_proba(U)[:,1]

# Logistic regression, cross entropy loss with logistic function
# Regularization here is commented out
# https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
# https://web.stanford.edu/~jurafsky/slp3/5.pdf (Eq. 5.17)
class UnregularizedLogisticRegressionClassifier(Classifier):
	def compute_theta(U, R, L_prime):
		learning_rate = 1e-0
		num_iter = 100

		# Initialize theta
		theta = np.zeros((U.shape[1], 1))

		def compute_gradient(X, t, w):
			t = t.reshape((-1, 1))
			assert t.shape[1] == 1
			assert t.shape[0] == X.shape[0]
			assert w.shape[0] == X.shape[1]
			assert w.shape[1] == 1

			# Compute intermediate values
			z = np.matmul(X, w)
			t_as_zero_or_one = 0.5 * t + 0.5
			
			gradient = np.matmul(np.transpose(X), LinAlg.logistic(z) - t_as_zero_or_one)
			return gradient

		# Gradient descent
		last_sum_abs_grad = None
		for it in range(num_iter):
			gradient = compute_gradient(U, L_prime, theta)
			theta = theta - learning_rate / len(U) * gradient
			
			# # If you want to add regularization, uncomment this line of code
			# # Higher L2_REGULARIZATION_COEFFICIENT increases regularization strength
			# L2_REGULARIZATION_COEFFICIENT = 1e-5
			# theta = theta - L2_REGULARIZATION_COEFFICIENT * theta
			
			# Check stopping condition
			NECESSARY_IMPROVEMENT_TO_CONTINUE = 0.1
			sum_abs_grad = np.sum(np.abs(gradient))
			if last_sum_abs_grad is not None:
				if last_sum_abs_grad - NECESSARY_IMPROVEMENT_TO_CONTINUE < sum_abs_grad: break
			last_sum_abs_grad = sum_abs_grad

		return theta
	
	def compute_scores(theta, U, R):
		return LinAlg.logistic(np.matmul(U, theta))

# Perceptron algorithm, Bishop 4.1.7
# Bishop Textbook: https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
class PerceptronClassifier(Classifier):
	def compute_theta(U, R, L_prime):
		learning_rate = 1e-4
		num_iter = 1
		
		# Initialize theta
		theta = np.zeros((U.shape[1], 1))

		def compute_gradient(X, t, w):
			t = t.reshape((-1, 1))
			assert t.shape[1] == 1
			assert t.shape[0] == X.shape[0]
			assert w.shape[0] == X.shape[1]
			assert w.shape[1] == 1

			# Compute intermediate values
			z = np.matmul(X, w)
			labels = sign(z)
			misclassified = (labels != t).astype(int)
			active_t = misclassified * t
			
			gradient = -np.matmul(np.transpose(X), active_t)
			return gradient
		
		# Gradient descent
		for _ in range(num_iter):
			gradient = compute_gradient(U, L_prime, theta)
			theta = theta - learning_rate / len(U) * gradient

		return theta
	
	def compute_scores(theta, U, R):
		return np.matmul(U, theta)
