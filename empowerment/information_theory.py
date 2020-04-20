"""
Information Theory 

This module contains functions for computing information-theoretic quantities on discrete probability distributions. Includes computation of 

- Entropy 
- Conditional Entropy
- KL Divergence 
- Mutual Information 
- Channel Capacity (using the Blahut-Arimoto algorithm)

 """

import numpy as np 
from numpy import log2

eps = 1e-40

def H(p_x):
    """
    Compute the entropy of a random variable distributed ~ p(x)
    """
    return -np.sum(p_x*log2(p_x))

def H_cond(P_yx, p_x):
    """ 
    Compute the conditional entropy H(Y|X) given distributions p(y|x) and p(x)
    """
    return -np.sum((P_yx*log2(P_yx))@p_x)

def KL_div(p_x, q_x):
    """
    Compute the KL-divergence between two random variables
    D(p||q) = E_p[log(p(X)/q(X))]

    p_x : defines p(x), shape (dim_X, )  
    q_x : defines q(x), shape (dim_X, )
    """
    if p_x.shape != q_x.shape:
        raise ValueError("p_x and q_x should have the same length")
    return np.sum(p_x * log2(p_x/q_x))

def I(Pxy):
    """
    Compute the mutual information between two random variables related by 
    their joint distribution p(x,y)

    Pxy : defines the joint distribution p(x,y), shape (dim_Y, dim_X)
    """
    p_x = np.sum(Pxy, axis = 0)
    p_y = np.sum(Pxy, axis = 1)
    return KL_div(Pxy, product(p_x, p_y))

def I2(P_yx, p_x):
    """
    Compute the mutual information between two random variables given the conditional distribution
    p(y|x) and p(x)
    
    P_yx : matrix defining p(y|x), shape (dim_Y, dim_X)  
    p_x :  defines distribution p(x), shape (dim_X,) 
    """
    p_y = (P_yx@p_x).reshape(-1,1)
    Pxy = P_yx/p_y
    return np.sum( (P_yx*log2(Pxy)) @ p_x )

def product(p_x, p_y):
    """
    Compute the product distribution p(x,y) = p(x)p(y) given distributions 
    p(x) and p(y)
    """
    p_x = p_x.reshape(1,-1)
    p_y = p_y.reshape(-1,1)
    Pxy = p_y@p_x
    assert Pxy.shape == (p_y.shape[0], p_x.shape[1]) 
    return Pxy

def blahut_arimoto(P_yx, epsilon = 0.001, deterministic = False):
    """ 
    Compute the channel capacity C of a channel p(y|x) using the Blahut-Arimoto algorithm. To do
    this, finds the input distribution p(x) that maximises the mutual information I(X;Y)
    determined by p(y|x) and p(x).

    P_yx : defines the channel p(y|x)
    iters : number of iterations
    """
    P_yx = P_yx + eps
    if not deterministic:
        # initialize input dist randomly 
        q_x = _rand_dist((P_yx.shape[1],))
        T = 1
        while T > epsilon:
            # update PHI
            PHI_yx = (P_yx*q_x.reshape(1,-1))/(P_yx @ q_x).reshape(-1,1)
            r_x = np.exp(np.sum(P_yx*log2(PHI_yx), axis=0))
            # channel capactiy 
            C = log2(np.sum(r_x))
            # check convergence 
            T = np.max(log2(r_x/q_x)) - C
            # update q
            q_x = _normalize(r_x + eps)
        if C < 0:
            C = 0
        return C
    else:
        # assume all columns in channel matrix are peaked on a single state
        # log of number of reachable states
        return log2(np.sum(P_yx.sum(axis=1) > 0.999))

def _rand_dist(shape):
    """ define a random probability distribution """
    P = np.random.rand(*shape)
    return _normalize(P)

def _normalize(P):
    """ normalize probability distribution """
    s = sum(P)
    if s == 0.:
        raise ValueError("input distribution has sum zero")
    return P / s
