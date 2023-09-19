import pandas as pd
import numpy as np

__all__ = ["compute_proportions"]

def compute_proportions(bags, y):
    num_bags = int(max(bags)) + 1
    proportions = np.empty(num_bags, dtype = np.float)
    for i in range(num_bags):
        bag = np.where(bags == i)[0]
        proportions[i] = np.count_nonzero(y[bag] == 1) / len(bag)
    return proportions
