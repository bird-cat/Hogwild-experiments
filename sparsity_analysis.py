#!/usr/bin/python3

import sys

def get_sparsity(input):
    non_zero_indices = 0
    n_samples = 0
    feature_sz = 0
    with open(input, "r") as f1:
        for line in f1:
            n_samples += 1
            split_line = line.split(' ')
            non_zero_indices += len(split_line) - 1
            feature_sz = max(feature_sz, len(split_line) - 1)
    return non_zero_indices / (n_samples * feature_sz)
            

print("sparsity = ", get_sparsity(sys.argv[1]))
