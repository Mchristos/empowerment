import numpy as np
import matplotlib.pyplot as plt 
import itertools
from functools import reduce
from empowerment.information_theory import blahut_arimoto
from empowerment import compute_empowerment

class MazeWorld(object):
    """ Represents an n x m grid world with walls at various locations. Actions can be performed (N, S, E, W, "stay") moving a player around the grid world. You can't move through walls. """

    def __init__(self, height, width, toroidal = False):
        """ 
        height : int 
            Height of grid world 
        width : int
            Width of grid world 
        toroidal: 
            If true, player can move off the edge of the world, appearing on the other side.   
        """
        self.dims = [height, width]
        self.height = height
        self.width = width
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

    def compute_model(self, det = 1.):
        """ Computes probabilistic model T[s',a,s] corresponding to the maze world. """
        n_actions = len(self.actions)
        n_states = self.dims[0]*self.dims[1]
        # compute environment dynamics as a matrix T 
        T = np.zeros([n_states,n_actions,n_states])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
        for s in range(n_states):
            for i, a in enumerate(self.actions.keys()):
                s_new = self.act(s, a)
                s_unc = list(map(lambda x : self.act(s, x), filter(lambda x : x != a, self.actions.keys())))
                T[s_new, i, s] += det
                for su in s_unc:
                    T[su, i, s] += (1-det)/(len(s_unc))
        self.T = T
        return T 

    def compute_empowerment(self, n_step, n_samples = 5000, det = 1.):
        """ 
        Computes the empowerment of each cell and returns as array. 

        n_step : int
            Determines the "time horizon" of the empowerment computation. The computed empowerment is the influence the agent has on the future over an n_step time horizon.
        n_samples : int
            Number of samples to use for sparse sampling empowerment computation (only used if the number of n-step actions exceeds 1000).         
        det : float between 0 and 1
            Probability of action successfully performed (otherwise a random different action is performed with probability 1 - det). When det = 1 the dynamics are deterministic. 
        """
        T = self.compute_model()
        E = np.zeros(self.dims)
        for y in range(self.dims[0]):
            for x in range(self.dims[1]):
                s = self._cell_to_index((y,x))
                E[y,x] = compute_empowerment(T=T, det = (det == 1.), n_step = n_step, state=s, n_samples=n_samples)
        return E

def klyubin_world():
    """ Build mazeworld from Klyubin et al. """
    maze = MazeWorld(10,10)
    # wall A 
    for i in range(6):
        maze.add_wall( (1, i), "N" )
    # wall B & D 
    for i in range(2):
        maze.add_wall( (i+2, 5), "E")
        maze.add_wall( (i+2, 6), "E")
    # wall C
    maze.add_wall( (3, 6), "N")
    # wall E
    for i in range(2):
        maze.add_wall( (1, i+7), "N")
    # wall F 
    for i in range(3):
        maze.add_wall( (5, i+2), "N")
    # wall G 
    for i in range(2):
        maze.add_wall( (i+6, 5), "W")
    # walls HIJK
    maze.add_wall( (6, 4), "N")
    maze.add_wall( (7, 4), "N")
    maze.add_wall( (8, 4), "W")
    maze.add_wall( (8, 3), "N")
    return maze

def door_world():
    """ Grid world used in Experiment 2 """
    maze = MazeWorld(height= 6, width = 9)
    for i in range(maze.dims[0]):
        if i is not 3:
            maze.add_wall( (i, 6), "W")
    for j in range(6):
        if j is not 0:
            maze.add_wall( (2 , j), "N")
    maze.add_wall((2,2), "E")
    maze.add_wall((0,2), "E")
    return maze

def tunnel_world():
    """ Grid world used in Experiment 3 """
    maze = MazeWorld(height= 5, width = 9)
    # vertical walls 
    for i in range(maze.dims[0]):
        if i is not 2:
            maze.add_wall( (i, 6), "W")
            maze.add_wall( (i, 2), "W")
    # tunnel walls 
    for j in range(2,6):
            maze.add_wall( (2 , j), "N")
            maze.add_wall( (2, j), "S")
    return maze

def step_world():
    n = 7
    maze = MazeWorld(n,n, toroidal=False)
    centre = np.array([(n-1)/2]*2, dtype = int)
    for i in (centre + [-1,+1]):
        for d in ['W','E']:
            maze.add_wall([i,centre[1]], d)
    for j in (centre + [-1,+1]):
        for d in ['N','S']:
            maze.add_wall([centre[0],j], d)
    return maze
