

"""
Like KDDWithSortingNetwork.py except that merge sort is used when computing
thresholds.

This file mainly exists to see if there is performance to be gained by using
merge sort. The performance benefit is at least a 50% increase in speed.

Computing thresholds with merge sort is a work in progress. This file does not
keep the user data private because there is currently no oblivious shuffle
performed.
"""

import time

import torch
import crypten
crypten.init()

import crypten.communicator as comm
import crypten.mpc as mpc

import random
import sys

import compute_server_io
import Workspace

def compute_theta_enc(U_enc, L_prime_enc):
    theta_initial_enc = crypten.cryptensor(torch.zeros((U_enc.shape[1], 1)))
    theta_enc = theta_initial_enc
    
    def compute_gradient_enc(X_enc, t_enc, w_enc):
        # Forward pass
        scores_enc = X_enc.matmul(w_enc).sigmoid()
        
        t_enc = t_enc.reshape((-1, 1))
        t_as_zero_or_one_enc = (t_enc * 0.5) + 0.5
        
        # Backward pass
        gradient_enc = X_enc.t().matmul(scores_enc - t_as_zero_or_one_enc) / X_enc.shape[0]
        
        # Apply L2 regularization
        gradient_enc = gradient_enc + Workspace.LOGISTIC_REGRESSION_L2_COEFFICIENT * w_enc
        
        # Return gradient
        return gradient_enc
    
    # Train
    for i in range(Workspace.LOGISTIC_REGRESSION_ITERATIONS):
        gradient_enc = compute_gradient_enc(U_enc, L_prime_enc, theta_enc)
        theta_enc = theta_enc - (Workspace.LOGISTIC_REGRESSION_LEARNING_RATE * gradient_enc)
    
    # Return learned weights
    return theta_enc

def compute_scores_enc(theta_enc, U_enc):
    return U_enc.matmul(theta_enc)

def compute_thresholds_enc(scores_enc, R, B):
    thresholds_enc = crypten.cryptensor(torch.zeros_like(B))
    for r in range(len(B)):
        start = time.time()
    
        # Collect indices
        indices_of_bag = torch.tensor([i for i in range(len(R)) if R[i] == r])

        # Compute k
        num_positive = int(round(B[r].item() * len(indices_of_bag)))
        k = num_positive
        k = max(0, k)
        k = min(len(indices_of_bag) - 1, k)
        
        # Extract kth least
        scores_of_bag_enc = scores_enc.view((-1,)).gather(0, indices_of_bag)
        
        # This measures the number of oblivious comparisons made and is used to evaluate the runtime of the algorithm.
        number_of_wires = 0
        
        # left and right are indices.
        # The range of [left, right] has to be sorted.
        def merge_sort(res, left, right):
            to_sort, number_of_wires = res
            if left > right: return (to_sort, number_of_wires)
            if left == right: return (to_sort, number_of_wires)
            
            mid = (left + right + 1) // 2
            
            (to_sort, number_of_wires) = merge_sort((to_sort, number_of_wires), left, mid - 1)
            (to_sort, number_of_wires) = merge_sort((to_sort, number_of_wires), mid, right)
            
            new_order = []
            left_index = left
            right_index = mid
            
            while left_index < mid and right_index <= right:
                comparison_one = to_sort[left_index] <= to_sort[right_index]
                # comparison_two = to_sort[right_index] <= to_sort[left_index]
                # comparison_three = to_sort_2[left_index] <= to_sort_2[right_index]
                use_left = comparison_one
                number_of_wires += 1
                use_left = use_left.get_plain_text()
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
            
            out = to_sort.clone()
            for i, index in enumerate(new_order):
                out[left + i] = to_sort[index]
            return (out, number_of_wires)
            
        scores_of_bag_enc, number_of_wires = merge_sort((scores_of_bag_enc, 0), 0, scores_of_bag_enc.shape[0] - 1)
        
        if comm.get().get_rank() == 0:
            print("In compute_thresholds_enc, voting region {:d} had {:d} people and performed {:d} oblivious comparisons.".format(r + 1, scores_of_bag_enc.shape[0], number_of_wires))
            print('Computing the threshold of that voting region took {:.2f} secodns.'.format(time.time() - start))
        
        # Save threshold in thresholds_enc
        thresholds_enc[r] = scores_of_bag_enc[k]

    return thresholds_enc

def algorithm_2(R, B):
    out = ((torch.gather(B, 0, R) >= 0.5).int() * 2) - 1
    return out 

def algorithm_1_enc(U_enc, R, B):
    if comm.get().get_rank() == 0:
        print('U_enc.shape is {!s:s}'.format(U_enc.shape))
        print('R.shape is {!s:s}'.format(R.shape))
        print('B.shape is {!s:s}'.format(B.shape))
    
    # Line 1 - Init labels
    start = time.time()
    L_prime = algorithm_2(R, B)
    L_prime_enc = crypten.cryptensor(L_prime)
    end = time.time()
    if comm.get().get_rank() == 0:
        print('Initting labels took {:.2f} seconds.'.format(end - start))
    
    # Line 2 - for loop
    # Line 3 removed due to MPC environment
    for iteration_number in range(Workspace.ALGORITHM_ONE_ROUNDS):
        if comm.get().get_rank() == 0:
            print()
            print('Beginning Algorithm 1 loop iteration {:d} / {:d}'.format(iteration_number + 1, Workspace.ALGORITHM_ONE_ROUNDS))

        # Line 4 - Fit logistic regression
        start = time.time()
        theta_enc = compute_theta_enc(U_enc, L_prime_enc)
        end = time.time()
        if comm.get().get_rank() == 0:
            print('Fitting LR took {:.2f} seconds.'.format(end - start))
        
        # Line 5 and 6 - Compute new thresholds
        start = time.time()
        scores_enc = compute_scores_enc(theta_enc, U_enc)
        end = time.time()
        if comm.get().get_rank() == 0:
            print('Computing scores took {:.2f} seconds.'.format(end - start))

        start = time.time()
        thresholds_enc = compute_thresholds_enc(scores_enc, R, B)
        end = time.time()
        if comm.get().get_rank() == 0:
            print('Computing thresholds took {:.2f} seconds.'.format(end - start))
        
        # Line 7 and 8 - Set new labels
        g_start = time.time()
        
        g = thresholds_enc.gather(0, R)
        g_end = time.time()
        
        c = scores_enc >= g
        c_end = time.time()
        
        m = c * 2
        m_end = time.time()
        
        L_prime_enc = m - 1
        s_end = time.time()
        
        if comm.get().get_rank() == 0:
            print('Setting labels took {:.2f} seconds.'.format(s_end - g_start))
            print('    Gather(enc, public) took {:.2f} seconds.'.format(g_end - g_start))
            print('    ge took {:.2f} seconds.'.format(c_end - g_end))
            print('    mul(enc, public) took {:.2f} seconds.'.format(m_end - c_end))
            print('    add(enc, public) took {:.2f} seconds.'.format(s_end - m_end))
    
    print('Algorithm 1 is done')
    # torch.save(thresholds_enc.share, '../thresholds_enc.pt')
    # torch.save(theta_enc.share, '../weights_enc.pt')
    # print('weights and thresholds are saved')
    print('TODO: save weights and thresholds')

num_users = 1000
num_websites = 60
num_regions = 51
U_enc, R, B = compute_server_io.make_fake_data(num_users, num_websites, num_regions)
algorithm_1_enc(U_enc, R, B)
