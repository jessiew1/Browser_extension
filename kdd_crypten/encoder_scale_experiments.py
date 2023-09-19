

# This program plays around with encoder scale and the underlying shares of cryptensors.

import torch
import crypten
crypten.init()

import crypten.communicator as comm
import crypten.mpc as mpc

ALICE = 0
BOB = 1

import random
import sys

def a():
    rank = comm.get().get_rank()
    print(f'Rank {rank}')
    k = 100
    alice_action = random.randint(0, k)
    bob_action = random.randint(0, k)
    if rank == ALICE: print(f'Rank {rank} action {alice_action}')
    if rank == BOB: print(f'Rank {rank} action {bob_action}')
    alice_action_enc = crypten.cryptensor([alice_action], src = ALICE)
    bob_action_enc = crypten.cryptensor([bob_action], src = BOB)
    result_enc = alice_action_enc + bob_action_enc
    result = result_enc.get_plain_text()
    print(f'Rank {rank} result {result}')
    
    print(dir(alice_action_enc))
    print(dir(alice_action_enc.encoder))
    print(alice_action_enc.encoder.scale)
    print(result_enc.encoder.scale)
    charlie_action_enc = crypten.cryptensor(torch.zeros((4, 1)), src = ALICE)
    ca = random.randint(0, 1000)
    print(f'{rank} ca {ca}')
    charlie_action_enc.share[0] += ca << 16
    print(charlie_action_enc.share)
    print(charlie_action_enc.get_plain_text())
    
print('before a')
sys.stdout.flush()
a()
print('after a')
sys.stdout.flush()
