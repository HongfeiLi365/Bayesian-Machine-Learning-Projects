# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:49:14 2018

@author: HL
"""
import numpy as np
from scipy.special import digamma

def log_gamma(x):
    z=1/(x*x)

    x=x+6;
    z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x;
    z=(x-0.5)*np.log(x)-x+0.918938533204673+z-np.log(x-1)-np.log(x-2)-np.log(x-3)-np.log(x-4)-np.log(x-5)-np.log(x-6);
    return z

def log_sum(log_a, log_b):
    if (log_a < log_b):
        v = log_b+np.log(1 + np.exp(log_a-log_b));
    else:
        v = log_a+np.log(1 + np.exp(log_b-log_a));
    return(v)

def trigamma(x):
    x=x+6;
    p=1/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)*p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for i in range(6):
        x=x-1;
        p=1/(x*x)+p;
    return(p);

'''
alpha
'''

def alhood(a,ss,D, K):
    return(D * (log_gamma(K * a) - K * log_gamma(a)) + (a - 1) * ss)

def d_alhood(a, ss, D, K):
    return(D * (K * digamma(K * a) - K * digamma(a)) + ss)


def d2_alhood(a, D,  K):
    return(D * (K * K * trigamma(K * a) - K * trigamma(a)))


def opt_alpha(ss,  D,  K, NEWTON_THRESH = 1e-5, MAX_ALPHA_ITER = 1000):
    init_a = 100
    i_iter = 0
    log_a = np.log(init_a)
    df = np.abs(NEWTON_THRESH + 1)
    
    while True:
        i_iter=i_iter + 1
        a = np.exp(log_a);
        if np.isnan(a):
            init_a = init_a * 10;
            print("warning : alpha is nan; new init = {}\n".format(init_a))
            a = init_a;
            log_a = np.log(a);
        f = alhood(a, ss, D, K)
        df = d_alhood(a, ss, D, K)
        d2f = d2_alhood(a, D, K)
        log_a = log_a - df/(d2f * a + df)
        if i_iter % 100 == 0: print("alpha maximization : {0:.5f}   {1:.5f}".format(f, df))
        if ((np.abs(df) < NEWTON_THRESH) or (i_iter >= MAX_ALPHA_ITER)): break
    
    return np.exp(log_a)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


