""" Module allowing for the computation of n-step empowerment given a matrix describing the probabilistic dynamics of an environment. """

import numpy as np 
from functools import reduce
import itertools
from info_theory import blahut_arimoto
import random

## CONSTANTS 
eps = 1e-30

## FUNCTIONS 
def rand_sample(p_x):
    """ randomly sample value from probability distribution """
    cumsum = np.cumsum(p_x)
    rnd = np.random.rand()
    return np.argmax(cumsum > rnd)

def normalize(X):
    """ normalize vector or matrix columns """
    return X / X.sum(axis=0)

def softmax(x, tau):
    return normalize(np.exp(x / tau)) 

def empowerment(T, det, n_step, state, n_samples=1000, epsilon = 1e-6):
    """ 
    T : numpy array, shape (n_states, n_actions, n_states)
        Transition matrix describing the probabilistic dynamics of a markov decision process (without rewards). Taking action a in state s, T describes a probability distribution over the resulting state as T[:,a,s]. In other words, T[s',a,s] is the probability of landing in state s' after taking action a in state s. The indices may seem "backwards" because this allows for convenient matrix multiplication.   
    det : bool
        True if the dynamics are deterministic.
    n_step : int 
        Determines the "time horizon" of the empowerment computation. The computed empowerment is the influence the agent has on the future over an n_step time horizon. 
    n_samples : int
        Number of samples for approximating the empowerment in the deterministic case.
    state : int 
        State for which to compute the empowerment.
    """
    n_states, n_actions, _  = T.shape
    if det:
        # only sample if too many actions sequences to iterate through
        if n_actions**n_step < 5000:
            nstep_samples = np.array(list(itertools.product(range(n_actions), repeat = n_step)))
        else:
            nstep_samples = np.random.randint(0,n_actions, [n_samples,n_step] )
        # fold over each nstep actions, get unique end states
        tmap = lambda s, a : np.argmax(T[:,a,s]) 
        seen = set()
        for i in range(len(nstep_samples)):
            aseq = nstep_samples[i,:]
            seen.add(reduce(tmap, [state,*aseq]))
        # empowerment = log # of reachable states 
        return np.log2(len(seen))
    else:
        nstep_actions = list(itertools.product(range(n_actions), repeat = n_step))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        for i, an in enumerate(nstep_actions):
            Bn[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
        return blahut_arimoto(Bn[:,:,state], epsilon=epsilon)




class EmpMaxAgent:
    """ Model-based RL agent maximising empowerment 
    
    This agent uses an intrinsic reinforcement learning framework to maximise its empowerment. It does this by computing its own reward signal, as the empowerment with respect to the current learned world model. 

    """
    def __init__(self, T, det, alpha = 0.1, gamma = 0.9, n_step=1, n_samples = 500):
        """ 
        T : transition matrix, numpy array
            Transition matrix describing environment dynamics. T[s',a,s] is the probability to transitioning to state s' given you did action a in state s. 
        alpha : learning rate, float
        gamma : discount factor, float 
            Discounts future rewards. Fraction, between 0 and 1.
        n_step : time horizon, int 
        n_samples : int
            Number of samples used in empowerment computation if environment is deterministic.    
        """
        self.alpha = alpha 
        self.gamma = gamma
        # transition function 
        self.T = T
        self.n_s, self.n_a, _ = T.shape
        # reward function 
        self.R = 5*np.ones([self.n_s,self.n_a])
        # experience 
        self.D = np.zeros([self.n_s,self.n_a,self.n_s])
        # action-value
        self.Q = 50*np.ones([self.n_s,self.n_a])
        self.n_step = n_step
        self.n_samples = n_samples
        self.det = det
        self.a_ = None
        self.k_s = 10
        self.k_a = 3
        self.E = np.array([self.estimateE(s) for s in range(self.n_s)])
        # temperature parameter 
        self.tau0 = 15 # initial tau
        self.tau = self.tau0 
        self.t = 0 
        self.decay = 5e-4

    def act(self, s):
        if self.a_ is None:
            return rand_sample(softmax(self.Q[s,:], tau=self.update_tau()))
        else:
            return self.a_
    
    def update_tau(self): 
        self.tau = self.tau0*np.exp(-self.decay*self.t)
        self.t += 1 
        return self.tau

    def update(self, s,a,s_):
        # append experience, update model
        self.D[s_,a,s] += 1  
        self.T[:,a,s] = normalize(self.D[:,a,s])
        # compute reward as empowerment achieved
        r = self.estimateE(s_)
        self.E[s_] = r
        # update reward R
        self.R[s,a] = self.R[s,a] + self.alpha*(r - self.R[s,a])
        # update action-value Q
        nz = np.where(self.T[:,a,s] != 0)[0]
        self.Q[s,a] = self.R[s,a] + self.gamma*(np.sum(self.T[nz,a,s]*np.max(self.Q[nz,:],axis=1)))
        # randomly update Q at other states 
        for s in np.random.randint(0,self.n_s, self.k_s):
            for a in np.random.randint(0,self.n_a, self.k_a):
                self.Q[s,a] = self.R[s,a] + self.gamma*(np.sum(self.T[:,a,s]*np.max(self.Q,axis=1)))
    
    def estimateE(self, state):
        return empowerment(self.T, self.det, self.n_step, state, n_samples = self.n_samples)