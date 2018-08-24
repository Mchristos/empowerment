import numpy as np
import matplotlib.pyplot as plt 
import itertools
from info_theory import blahut_arimoto
from functools import reduce



def empowerment(maze, n_step = 5, det = 1., n_samples = 5000):
    actions = {
        0 : "N",
        1 : "S",
        2 : "E",
        3 : "W",
        4 : "_" 
    }
    n_actions = len(actions)
    n_states = maze.dims[0]*maze.dims[1]
    B = np.zeros([n_states,n_actions,n_states])
    for s in range(n_states):
        for a in actions.keys():
            s_new = maze.act(s, actions[a])
            s_unc = list(map(lambda x : maze.act(s, actions[x]), filter(lambda x : x != a, actions.keys())))
            B[s_new, a, s] += det
            for su in s_unc:
                B[su, a, s] += (1-det)/(len(s_unc))
    def empowerment(state, det):
        if det == 1.:
            nstep_samples = np.random.randint(0,n_actions, [n_samples,n_step] )
            # fold over each nstep actions, get unique end states
            tmap = lambda s, a : np.argmax(B[:,a,s]) 
            seen = set()
            for i in range(len(nstep_samples)):
                aseq = nstep_samples[i,:]
                seen.add(reduce(tmap, [state,*aseq]))
            return np.log2(len(seen))
        else:
            nstep_actions = list(itertools.product(range(n_actions), repeat = n_step))
            Bn = np.zeros([n_states, len(nstep_actions), n_states])
            for i, an in enumerate(nstep_actions):
                Bn[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : B[:,a,:]), an))
            return blahut_arimoto(Bn[:,:,state], epsilon=1e-10)
    E = np.zeros(maze.dims)
    for y in range(maze.dims[0]):
        for x in range(maze.dims[1]):
            s = maze._cell_to_index((y,x))
            E[y,x] = empowerment(s, det)
    return E
    

class MazeWorld(object):
    """ Represents an nxm maze """

    def __init__(self, height, width, toroidal = False):
        self.dims = [height, width]
        self.adjacencies = dict()
        self.actions = {
            "N" : np.array([1, 0]), # UP
            "S" : np.array([-1,0]),  # DOWN
            "E" : np.array([0, 1]), # RIGHT
            "W" : np.array([0,-1]), # LEFT
            "_" : np.array([0, 0])  # STAY
        }
        self.opposite = {
            "N" : "S",
            "S" : "N",
            "W" : "E",
            "E" : "W"
        }
        for i in range(height):
            self.adjacencies[i] = dict()
            for j in range(width):
                self.adjacencies[i][j] = list(self.actions.keys())
        self.walls = []
        self.toroidal = toroidal
        self.vecmod = np.vectorize(lambda x, y : x % y)

    def add_wall(self, cell, direction):
        cell = np.array(cell)
        # remove action 
        self.adjacencies[cell[0]][cell[1]].remove(direction)
        # remove opposite action
        new_cell = cell + self.actions[direction]
        self.adjacencies[new_cell[0]][new_cell[1]].remove(self.opposite[direction])
        # save wall for plotting 
        self.walls.append((cell, new_cell))

    def act(self, s, a, prob = 1.):
        """ get updated state after action
    
        s  : state, index of grid position 
        a : action 
        prob : probability of performing action
        """
        rnd = np.random.rand()
        if rnd > prob:
            a = np.random.choice(list(filter(lambda x : x !=a, self.actions.keys())))
        state = self._index_to_cell(s)
        if a not in self.adjacencies[state[0]][state[1]]:
            return self._cell_to_index(state)
        new_state = state + self.actions[a] 
        # can't move off grid
        if self.toroidal:
            new_state = self.vecmod(new_state, self.dims)
        else:
            if np.any(new_state < np.zeros(2)) or np.any(new_state >= self.dims):
                return self._cell_to_index(state)
        return self._cell_to_index(new_state)
    
    def act_nstep(self, s, actions):
        for a in actions:
            s = self.act(s, a)
        return s

    def _index_to_cell(self, s):
        cell = [int(s / self.dims[1]), s % self.dims[1]]
        return np.array(cell)

    def _cell_to_index(self, cell):
        return cell[1] + self.dims[1]*cell[0]
    
    def plot(self, pos = None, traj = None, action = None, colorMap = None, vmin = None, vmax = None):
        G = np.zeros(self.dims) if colorMap is None else colorMap.copy()
        # plot color map 
        if vmax is not None:
            plt.pcolor(G, vmin = vmin, vmax=vmax) # , cmap = 'Greys')
        else:
            plt.pcolor(G)
        plt.colorbar()
        if pos is not None:
            plt.scatter([pos[1] + 0.5], [pos[0] + 0.5], s = 100, c = 'w')
        # plot trajectory
        if traj is not None:
            y, x = zip(*traj)
            y = np.array(y) + 0.5
            x = np.array(x) + 0.5
            plt.plot(x, y)
            plt.scatter([x[0]], [y[0]], s = 100, c = 'b')
            plt.scatter([x[-1]], [y[-1]], s = 100, c = 'r')
        for wall in self.walls:
            y, x = zip(*wall)
            if x[0] == x[1]:
                # horizontal wall 
                x = [x[0], x[0] + 1]
                y = [max(y), max(y)]
            else:
                # vertical wall
                x = [max(x), max(x)]
                y = [y[0], y[0] + 1]
            plt.plot(x, y, c = 'w')
        if action is not None:
            plt.title(str(action))


