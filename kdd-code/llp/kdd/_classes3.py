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

    def __init__(self, model, init_y = 'majority', max_iter=100):
        self.model = model
        self.max_iter = max_iter
        self.thresholds = None

        # Different scoring functions for SVM and LR.
        # For LR, we use the positive class probability
        # FOR SVM, we use the decision function score
        if type(self.model).__name__.endswith('SVC'):
            self.score_function = self.model.decision_function
        elif type(self.model).__name__.endswith('LogisticRegression'):
            self.score_function = lambda x: self.model.predict_proba(x)[:, 1]
        else:
            raise Exception('Invalid Model for EM based approach!')

        # which methods should be used to initialize y
        if init_y == 'majority':
            self.init_y = self._create_y_majority
        elif init_y == 'random':
            self.init_y = self._create_y
        else:
            raise Exception('Invalid init_y for EM based approach')

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_params(self):
        return self.__dict__

    def _create_y(self, bags, proportions):
        '''
        Purely random
        '''
        n = len(bags)
        return 2 * np.random.randint(2, size=n) - 1

    def _create_y_majority(self, bags, proportions):
        '''
        This was used in the KDD paper
        '''
        y = np.full(len(bags), -1)
        num_bags = len(proportions)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            if len(bag) != 0:
                if proportions[i] >= 0.5:
                    y[bag] = 1
                else:
                    y[bag] = -1
        return y

    def _create_y_bag_random(self, bags, proportions):
        '''
        Respect proportions within each bag, but randomly
        '''
        y = np.full(len(bags), -1)
        num_bags = len(proportions)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            if len(bag) != 0:
                I = np.random.choice(
                    bag, int(round(proportions[i] * len(bag))))
                y[I] = 1
        return y

    def predict(self, X, bags=None):
        if bags is None:
            return np.array(self.model.predict(X))
        else:
            rows, columns = X.shape
            y = np.full(rows, -1)
            #proba = self.model.predict_proba(X)[:, 1]
            #proba = self.model.decision_function(X)
            score = self.score_function(X)
            num_bags = int(max(bags) + 1)
            for i in range(num_bags):
                bag = np.where(bags == i)[0]
                if len(bag) != 0:
                    #print(i, bag, self.thresholds[i], self.thresholds)
                    I = [x for x in bag if score[x] >= self.thresholds[i]]
                    if len(I) != 0:
                        y[I] = 1
            return y

    def _optimize_y(self, score, bags, proportions):
        new_y = np.full(len(bags), -1)
        num_bags = len(proportions)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            if len(bag) != 0:
                bag_sorted = sorted(bag, key=lambda w: score[w], reverse=True)
                ones = int(round(proportions[i] * len(bag)))
                new_y[bag_sorted[:ones]] = 1
                new_y[bag_sorted[ones:]] = -1
                self.thresholds[i] = score[bag_sorted[ones - 1]]
        return new_y

    def fit(self, X, bags, proportions):

        self.thresholds = np.full(len(proportions), 0.5)

        #y = self._create_y_majority(bags, proportions)
        #y = self._create_y(len(bags))
        #y = self._create_y_bag_random(bags, proportions)
        #print('Sum and length of y, proportions', sum(y), len(y), compute_proportions(bags, y))
        y = self.init_y(bags, proportions)

        past_y = y.copy()
        past_error = None
        smallest_error = np.inf
        for it in range(self.max_iter):

            self.model.fit(X, y)
            #proba = self.model.predict_proba(X)[:, 1]
            #proba = self.model.decision_function(X)
            score = self.score_function(X)
            y, past_y = past_y, y
            y = self._optimize_y(score, bags, proportions)

            error = np.linalg.norm(y - past_y, ord=0)
            #print('Sum and length of y', sum(y), len(y), error)

            if past_error is not None:
                if abs(past_error - error) < 0.05 * len(y):
                    break

            past_error = error
