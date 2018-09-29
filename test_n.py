import numpy as np

def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

if __name__ == '__main__':
    l = [1,2,3,4,5,5,6,7,8,9]
    l_n = max_min_normalization(l)
    print(l_n)

