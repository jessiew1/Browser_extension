

"""
This program demonstrates saving and loading cryptensors by accessing its share
property and saving that to a file using torch.save.
"""

import time

import torch
import crypten
crypten.init()

import crypten.communicator as comm
import crypten.mpc as mpc

import random
import sys

try:
    # Pretend that cryptensor_to_save is a secret cryptensor that no compute party knows
    cryptensor_to_save = crypten.cryptensor(torch.randint(0, 0xFFFF, (1,)), src = 0)
    
    # Print and save
    print(comm.get().get_rank(), "cryptensor_to_save plaintext", cryptensor_to_save.get_plain_text())
    print(comm.get().get_rank(), "saved share", cryptensor_to_save.share)
    torch.save(cryptensor_to_save.share, '../e.pt')
    
    # Load shares from file
    share = torch.load('../e.pt')

    # Put loaded shares into new cryptensor
    loaded_cryptensor = crypten.cryptensor(torch.zeros(1,))
    loaded_cryptensor.share += share
    
    # Print results
    print(comm.get().get_rank(), "loaded share", share)
    print(comm.get().get_rank(), "loaded cryptensor plaintext", loaded_cryptensor.get_plain_text())
except Exception as e:
    print(e)
