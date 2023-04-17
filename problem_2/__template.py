import numpy as np

def solution(X, filters, stride=1, pad=0):
    h, w = X.shape
    f_h, f_w = filters.shape
    
    out_h = (h + 2 * pad - f_h) // stride + 1
    out_w = (w + 2 * pad - f_w) // stride + 1
 
    in_X = np.pad(X, [(pad, pad), (pad, pad)], 'constant')
    out = np.zeros((out_h, out_w))
    
    for i in range(h - f_h + 1):
        for j in range(w - f_w + 1):
            out[i, j] = np.sum(filters*in_X[i : i + f_h, j : j + f_w])
 
    return out


