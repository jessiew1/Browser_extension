

"""
This function prints the cost of using a sorting network versus merge sort.
The cost is measured by the number of comparisons made.
"""

import random
import sys

from sorting_network import sorting_network

def merge_sort_helper(res, left, right):
    to_sort, number_of_wires = res
    if left > right: return (to_sort, number_of_wires)
    if left == right: return (to_sort, number_of_wires)
    
    mid = (left + right + 1) // 2
    
    (to_sort, number_of_wires) = merge_sort_helper((to_sort, number_of_wires), left, mid - 1)
    (to_sort, number_of_wires) = merge_sort_helper((to_sort, number_of_wires), mid, right)
    
    new_order = []
    left_index = left
    right_index = mid
    
    while left_index < mid and right_index <= right:
        va = to_sort[left_index]
        vb = to_sort[right_index]
        use_left = 1 if va < vb else 0
        number_of_wires += 1
        assert use_left in [0, 1]
        if use_left == 1:
            new_order += [left_index]
            left_index += 1
        else:
            new_order += [right_index]
            right_index += 1
    while left_index < mid:
        new_order += [left_index]
        left_index += 1
    while right_index < right:
        new_order += [right_index]
        right_index += 1
    
    out = to_sort[:]
    for i, index in enumerate(new_order):
        out[left + i] = to_sort[index]
    return (out, number_of_wires)

def merge_sort_average(n):
    to_sort = list(range(n))
    random.shuffle(to_sort)
    out, number_of_wires = merge_sort_helper((to_sort, 0), 0, n - 1)
    return number_of_wires

def merge_sort_worst_helper(number_of_wires, left, right):
    if left > right: return number_of_wires
    if left == right: return number_of_wires
    
    mid = (left + right + 1) // 2
    
    number_of_wires = merge_sort_worst_helper(number_of_wires, left, mid - 1)
    number_of_wires = merge_sort_worst_helper(number_of_wires, mid, right)
    
    left_index = left
    right_index = mid
    
    while left_index < mid:
        number_of_wires += 1
        left_index += 1
    while right_index < right:
        number_of_wires += 1
        right_index += 1
    
    return number_of_wires

def merge_sort_worst(n):
    number_of_wires = merge_sort_worst_helper(0, 0, n - 1)
    return number_of_wires

def merge_sort_best_helper(res, left, right):
    to_sort, number_of_wires = res
    if left > right: return (to_sort, number_of_wires)
    if left == right: return (to_sort, number_of_wires)
    
    mid = (left + right + 1) // 2
    
    (to_sort, number_of_wires) = merge_sort_best_helper((to_sort, number_of_wires), left, mid - 1)
    (to_sort, number_of_wires) = merge_sort_best_helper((to_sort, number_of_wires), mid, right)
    
    new_order = []
    left_index = left
    right_index = mid
    
    while left_index < mid and right_index <= right:
        va = to_sort[left_index]
        vb = to_sort[right_index]
        use_left = 1 if va < vb else 0
        use_left = 1
        number_of_wires += 1
        assert use_left in [0, 1]
        if use_left == 1:
            new_order += [left_index]
            left_index += 1
        else:
            new_order += [right_index]
            right_index += 1
    while left_index < mid:
        new_order += [left_index]
        left_index += 1
    while right_index < right:
        new_order += [right_index]
        right_index += 1
    
    out = to_sort[:]
    for i, index in enumerate(new_order):
        out[left + i] = to_sort[index]
    return (out, number_of_wires)

def merge_sort_best(n):
    to_sort = list(range(n))
    random.shuffle(to_sort)
    out, number_of_wires = merge_sort_best_helper((to_sort, 0), 0, n - 1)
    return number_of_wires

def merge_sort_random_helper(res, left, right):
    to_sort, number_of_wires = res
    if left > right: return (to_sort, number_of_wires)
    if left == right: return (to_sort, number_of_wires)
    
    mid = (left + right + 1) // 2
    
    (to_sort, number_of_wires) = merge_sort_helper((to_sort, number_of_wires), left, mid - 1)
    (to_sort, number_of_wires) = merge_sort_helper((to_sort, number_of_wires), mid, right)
    
    new_order = []
    left_index = left
    right_index = mid
    
    while left_index < mid and right_index <= right:
        va = to_sort[left_index]
        vb = to_sort[right_index]
        use_left = 1 if va < vb else 0
        use_left = random.randint(0, 1)
        number_of_wires += 1
        assert use_left in [0, 1]
        if use_left == 1:
            new_order += [left_index]
            left_index += 1
        else:
            new_order += [right_index]
            right_index += 1
    while left_index < mid:
        new_order += [left_index]
        left_index += 1
    while right_index < right:
        new_order += [right_index]
        right_index += 1
    
    out = to_sort[:]
    for i, index in enumerate(new_order):
        out[left + i] = to_sort[index]
    return (out, number_of_wires)

def merge_sort_random(n):
    to_sort = list(range(n))
    random.shuffle(to_sort)
    out, number_of_wires = merge_sort_helper((to_sort, 0), 0, n - 1)
    return number_of_wires

def sorting_network_average(n):
    return len(list(sorting_network(n)))

NUM_TRIALS = 100

for k in range(1, 300):
    score_b_sum = 0
    score_m_sum = 0
    score_mr_sum = 0
    
    for _ in range(NUM_TRIALS):
        score_b_sum += sorting_network_average(k)
        score_m_sum += merge_sort_average(k)
        score_mr_sum += merge_sort_average(k)
        
    score_mw_sum = merge_sort_worst(k)
    score_mb_sum = merge_sort_best(k)
    
    print(
        'k {:d} b {:.2f} m {:.2f} mw {:d} mb {:d} mr {:.2f}'.format(
            k,
            score_b_sum / NUM_TRIALS,
            score_m_sum / NUM_TRIALS,
            score_mw_sum,
            score_mb_sum,
            score_mr_sum / NUM_TRIALS,
        ),
    )

    sys.stdout.flush()