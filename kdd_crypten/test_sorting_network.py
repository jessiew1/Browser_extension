from sorting_network import sorting_network, compare_and_swap

import random

def gen_input(n):
    out = list(range(n))
    random.shuffle(out)
    return out

def is_sorted(x):
    return x == sorted(x)

def test(n):
    """
    Test sorting_network on a random list of length n.
    """
    
    wires = list(sorting_network(n))
    assert isinstance(wires, list)
    for x in wires:
        assert isinstance(x, tuple)
        assert len(x) == 2
        low, high = x
        assert isinstance(low, int)
        assert isinstance(high, int)
        assert 0 <= low
        assert low < high
        assert high < n
    
    x = gen_input(n)
    for a, b in wires:
        compare_and_swap(x, a, b)
    
    assert is_sorted(x)

if __name__ == '__main__':
    # Run tests on sorting_network
    MAX_INPUT_LENGTH = 1 << 8
    NUM_TRIALS_PER_INPUT_LENGTH = 1 << 4
    for i in range(MAX_INPUT_LENGTH + 1):
        for j in range(NUM_TRIALS_PER_INPUT_LENGTH):
            test(i)
        print('Length {:8d} : {:d} / {:d} PASSED'.format(i, NUM_TRIALS_PER_INPUT_LENGTH, NUM_TRIALS_PER_INPUT_LENGTH))