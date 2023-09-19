# This code implements the algorithms from https://math.mit.edu/~shor/18.310/batcher.pdf.

def compare_and_swap(x, i, j) -> None:
    """
    If x[i] > x[j], then swap x[i] and x[j]
    """
    
    if x[i] > x[j]:
        x[i], x[j] = x[j], x[i]

def sorting_network(length: int):
    """
    Args:
        length (int): the length of the list to be sorted
    
    Yield:
        Pairs of indices to compare and swap if out of order.
            Indices begin at zero.
    """
    next_power_of_two = 1;
    while next_power_of_two < length:
        next_power_of_two *= 2
    
    for x in lecture_sort(1, next_power_of_two):
        (a, b) = x
        a -= 1
        b -= 1
        
        # Do not make a comparison if not both elements exist
        # This does not affect correctness because one can imagine padding the input with elements with value infinity and letting the omitted comparisons be comparisons with infinity.
        if b >= length:
            continue
        
        yield (a, b)

def lecture_sort(left, right):
    if left + 1 == right:
        # Base case: two elements to sort
        yield (left, right)
    elif left < right:
        # Recursive case: more than two elements to sort
        mid = (left + right) // 2
        yield from lecture_sort(left, mid)
        yield from lecture_sort(mid + 1, right)
        yield from lecture_merge(left, right, 1)

def lecture_merge(left: int, right: int, step: int):
    if left + 2 * step > right and left + step <= right:
        # Base case: two elements to merge
        yield (left, left + step)
    else:
        # Recursive case: more than two elements to merge
        if left + 2 * step <= right:
            yield from lecture_merge(left, right, step * 2)
            yield from lecture_merge(left + step, right, step * 2)
        for i in range(left + step, right - step, 2 * step):
            yield (i, i + step)
