""" Module allowing for the computation of n-step empowerment given a matrix describing the probabilistic dynamics of an environment. """

import numpy as np 
from functools import reduce
import itertools
from info_theory import blahut_arimoto

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