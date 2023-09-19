

# This program is the work-in-progress MPC version of the KDD paper's Algorithm 1.
# It uses the sorting network to compute thresholds.
# It generates fake data and then runs the algorithm on the fake data.
# It is to be run on three separate compute servers.
# It prints the time to complete each step.

import time

import torch
import crypten
crypten.init()

import crypten.communicator as comm
import crypten.mpc as mpc

import random
import sys

import compute_server_io
from sorting_network import sorting_network
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
        wires = sorting_network(len(indices_of_bag))
        number_of_wires = 0
        for a, b in wires:
            va = scores_of_bag_enc[a]
            vb = scores_of_bag_enc[b]
            to_swap = va > vb
            not_to_swap = 1 - to_swap
            scores_of_bag_enc[a] = to_swap * vb + not_to_swap * va
            scores_of_bag_enc[b] = to_swap * va + not_to_swap * vb
            
            number_of_wires += 1
        
        if comm.get().get_rank() == 0:
            print("In compute_thresholds_enc, voting region {:d} had {:d} people and performed {:d} oblivious comparisons.".format(r + 1, scores_of_bag_enc.shape[0], number_of_wires))
            print('Computing the threshold of that voting region took {:.2f} secodns.'.format(time.time() - start))
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
    print('TODO: save weights and thresholds')
    
    # # Save trained model parameters
    # torch.save(thresholds_enc.share, '../thresholds_enc.pt')
    # torch.save(theta_enc.share, '../weights_enc.pt')
    # print('weights and thresholds are saved')

num_users = 1000
num_websites = 60
num_regions = 51
U_enc, R, B = compute_server_io.make_fake_data(num_users, num_websites, num_regions)
algorithm_1_enc(U_enc, R, B)
