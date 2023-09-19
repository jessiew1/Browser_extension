

"""
This code is not tested.

Currently, it only shuffles one cryptensor, but what is needed is to shuffle two
cryptensors by the same permutation. The two cryptensors to be shuffled are the
actual values paired with unique values that break ties, so that the comparison
results do not leak information about the equality of the actual values.
"""

import time

import torch
import crypten
crypten.init()

import crypten.communicator as comm
import crypten.mpc as mpc

import random
import sys
import os

from Cryptodome.Cipher import AES

def int_to_bytes(x):
    out = []
    while x > 0:
        out = [x % 256] + out
        x //= 256
    while len(out) < 16:
        out = [0] + out
    return bytes(out)

def bytes_to_int(x):
    return int.from_bytes(x, "big", signed = True)

def PRP(k, x):
    assert isinstance(k, bytes)
    assert len(k) == 16
    assert isinstance(x, bytes)
    assert len(x) == 16
    
    y = AES.new(k, AES.MODE_ECB).encrypt(x)
    return y

def PRF(k, x):
    assert isinstance(k, bytes)
    assert len(k) == 16
    assert isinstance(x, bytes)
    assert len(x) == 16
    
    y = AES.new(k, AES.MODE_ECB).encrypt(x)
    y = y[:8]
    return y

def apply_keyed_permutation(key, input_vector):
    assert isinstance(key, bytes), type(key)
    assert len(key) == 16
    assert isinstance(input_vector, torch.Tensor)
    
    out = torch.clone(input_vector)
    
    x = [v for v in out]

    # Schwartzian transform
    x = [(i, x[i]) for i in range(len(x))]
    x = sorted(x, key = lambda v: PRP(key, int_to_bytes(v[0])))
    x = [v[1] for v in x]
    
    out = torch.tensor(x)
    
    assert out.shape == input_vector.shape
    
    return out

def generate_zero_scrambling(n, key):
    assert isinstance(n, int)
    assert isinstance(key, bytes)
    assert len(key) == 16
    
    out = torch.zeros(n, dtype = torch.long)
    
    for i in range(n):
        x = int_to_bytes(i)
        y = PRF(key, x)
        y = bytes_to_int(y)
        out[i] = y
    
    return out

def send_tensor(src, dst, message_shape, message):
    if comm.get().get_rank() == src:
        comm.get().send(message, dst = dst)
        
    if comm.get().get_rank() == dst:
        # NOTE: ignore share variable, what is received should be in the message_received variable.
        share = torch.zeros(message_shape, dtype = torch.long)
        message_received = comm.get().recv(share, src = src)
        return message_received
    else:
        return None

def send_bytes(src, dst, num_bytes, message):
    assert len(message) == num_bytes
    
    # Preprocess the message to be sent
    message_as_tensor = torch.tensor([int(x) for x in message])

    # Send message
    message_received = send_tensor(src, dst, message_as_tensor.shape, message_as_tensor)
    
    # Postprocess the received message
    if comm.get().get_rank() == dst:
        message_received = bytes([int(message_received[i]) for i in range(num_bytes)])
        return message_received
    else:
        return None

def oblivious_shuffle(input_enc):
    if len(input_enc.shape) != 1:
        raise NotImplementedError("Only 1-dimensional cryptensors are supported right now.")

    # Party numbers are 0, 1, 2
    # Shares are non-replicated 3-out-of-3 arithmetic sharings
    
    # Party 2 gives shares to party 1
    result = send_tensor(2, 1, input_enc.share.shape, input_enc.share)
    
    # Party 1 adds Party 2's shares to its own shares to form a 2-out-of-2 sharing.
    if comm.get().get_rank() == 1:
        input_enc.share = input_enc.share + result
    if comm.get().get_rank() == 2:
        input_enc.share[:] = 0
    
    # Party 0 generates PRP keys and sends to party 1
    key00 = os.urandom(16)
    key00_rec = send_bytes(0, 1, 16, key00)
    
    key01 = os.urandom(16)
    key01_rec = send_bytes(0, 1, 16, key01)
    
    # Parties 0 and 1 shuffle
    if comm.get().get_rank() == 0:
        input_enc.share = apply_keyed_permutation(key00, input_enc.share) + generate_zero_scrambling(input_enc.shape[0], key01)
    if comm.get().get_rank() == 1:
        input_enc.share = apply_keyed_permutation(key00_rec, input_enc.share) - generate_zero_scrambling(input_enc.shape[0], key01_rec)
    
    # Party 0 sends shares to party 2
    share2_rec = send_tensor(0, 2, input_enc.share.shape, input_enc.share)
    if comm.get().get_rank() == 2:
        input_enc.share = share2_rec
    
    # Party 1 generates PRP keys and sends to party 2
    key10 = os.urandom(16)
    key10_rec = send_bytes(1, 2, 16, key10)
    
    key11 = os.urandom(16)
    key11_rec = send_bytes(1, 2, 16, key11)
    
    # Parties 1 and 2 shuffle
    if comm.get().get_rank() == 1:
        input_enc.share = apply_keyed_permutation(key10, input_enc.share) + generate_zero_scrambling(input_enc.shape[0], key11)
    if comm.get().get_rank() == 2:
        input_enc.share = apply_keyed_permutation(key10_rec, input_enc.share) - generate_zero_scrambling(input_enc.shape[0], key11_rec)
    
    # Party 1 sends shares to party 0
    share3_rec = send_tensor(1, 0, input_enc.share.shape, input_enc.share)
    if comm.get().get_rank() == 0:
        input_enc.share = share3_rec
    
    # Party 2 generates PRP keys and sends to party 0
    key20 = os.urandom(16)
    key20_rec = send_bytes(2, 0, 16, key20)
    
    key21 = os.urandom(16)
    key21_rec = send_bytes(2, 0, 16, key21)
    
    # Parties 2 and 0 shuffle
    if comm.get().get_rank() == 2:
        input_enc.share = apply_keyed_permutation(key20, input_enc.share) + generate_zero_scrambling(input_enc.shape[0], key21)
    if comm.get().get_rank() == 0:
        input_enc.share = apply_keyed_permutation(key20_rec, input_enc.share) - generate_zero_scrambling(input_enc.shape[0], key21_rec)
    
    # Party 0 splits shares and sends split to party 1
    sent = torch.zeros(input_enc.shape, dtype = torch.long)
    if comm.get().get_rank() == 0:
        key4 = os.urandom(16)
        kept = torch.zeros(input_enc.shape, dtype = torch.long)
        kept = generate_zero_scrambling(input_enc.shape[0], key4)
        sent = input_enc.share - kept
        input_enc.share = kept
    received = send_tensor(0, 1, sent.shape, sent)
    if comm.get().get_rank() == 1:
        input_enc.share = received
    
    # Protocol is finished
    return input_enc

# Code to test script
x = torch.tensor([i for i in range(1000)])
x_e = crypten.cryptensor(x)
x_e = oblivious_shuffle(x_e)
print('rank {:d} x {!s:s}'.format(comm.get().get_rank(), x_e[:10].get_plain_text()))