import os
import sys
import random
import numpy as np
from copy import copy, deepcopy
import random

from statsmodels.distributions.empirical_distribution import ECDF
from llp.base import baseLLPClassifier
from abc import ABC, abstractmethod

class EM(baseLLPClassifier, ABC):

	def __init__(self, model):
		self.model = model
		self.t = None

	def set_params(self, **params):
		self.model.set_params(**params)

	def get_params(self):
		return self.__dict__

	# Algorithm 2 of KDD paper
	def _create_y_majority(self, R, B):
		L_prime = np.full(len(R), -1)
		for r in range(len(B)):
			bag = np.where(R == r)[0]
			if len(bag) == 0: continue
			L_prime[bag] = 1 if B[r] >= 0.5 else -1
		return L_prime

	def predict(self, X, R):
		y = np.full(X.shape[0], -1)
		P = self.model.predict_proba(X)[:, 1]
		for r in range(int(max(R) + 1)):
			bag = np.where(R == r)[0]
			if len(bag) == 0: continue
			#print(r, bag, self.t[r], self.t)
			above_threshold_in_bag = [x for x in bag if P[x] >= self.t[r]]
			if len(above_threshold_in_bag) == 0: continue
			y[above_threshold_in_bag] = 1
		return y

	def fit(self, U, R, B):
		# Line 1
		L_prime = self._create_y_majority(R, B)



		past_error = None
		
		
		
		# Line 2
		for it in range(100):
		
		
		
			# Line 3
			L_prime_prime = L_prime.copy()
			
			
			
			# Line 4
			self.model.fit(U, L_prime)
			
			
			
			# Line 5
			P = self.model.predict_proba(U)[:, 1]
			L_prime = np.full(len(R), -1)
			self.t = np.full(len(B), 0.5)
			for r in range(len(B)):
			
			
			
				# Line 6
				bag = np.where(R == r)[0]
				if len(bag) == 0: continue
				ones = int(round(B[r] * len(bag)))
				bag_sorted = sorted(bag, key=lambda w: P[w], reverse=True)
				self.t[r] = P[bag_sorted[ones - 1]]
				
				
				
				# Line 7 and 8
				L_prime[bag_sorted[:ones]] = 1
				L_prime[bag_sorted[ones:]] = -1



			# error is num mismatches of L_prime and L_prime_prime
			error = np.linalg.norm(L_prime - L_prime_prime, ord=0)
			# print('Sum and length of L\'', sum(L_prime), len(L_prime), error)

			# Line 9
			if past_error is not None:
				if abs(past_error - error) < 0.05 * len(L_prime):
					break
			past_error = error
		
		
		
		print('Model terminated in {:d} iterations'.format(it))