

# Contains functions for handling the inputs and outputs of the MPC computation.

import torch

import crypten
crypten.init()

def save_shares(encrypted_tensor, file_name):
    raise NotImplementedError("Raising error because the following code is untested")
    share = encrypted_tensor.share
    torch.save(share, file_name)

def load_shares(file_name):
    raise NotImplementedError("Raising error because the following code is untested")
    share = torch.load(file_name)
    
    encrypted_tensor = crypten.cryptensor(torch.zeros_like(shares))
    encrypted_tensor.share += share
    
    return encrypted_tensor

def make_fake_data(num_users, num_websites, num_regions):
    # Make some fake data
    U = torch.rand((num_users, num_websites))
    R = torch.randint(0, num_regions, (num_users,), dtype = torch.long)
    B = torch.rand((num_regions,))
    
    # Normalize U to make elements sum to 1
    U = U / torch.sum(U, 1, keepdim=True)
    
    # Slightly hacky way of ensuring that everyone gets the same copy of the fake data
    R = crypten.cryptensor(R, src = 0)
    B = crypten.cryptensor(B, src = 0)
    R = R.get_plain_text()
    B = B.get_plain_text()
    R = R.long()
    
    # Share the user web browsing data
    U_enc = crypten.cryptensor(U, src = 0)
    
    return (U_enc, R, B)
