import mazeworld
from mazeworld import MazeWorld
from empowerment import empowerment, EmpMaxAgent
import numpy as np 
import matplotlib.pyplot as plt
import time 

def example_1():      
    """ builds maze world from original empowerment paper(https://uhra.herts.ac.uk/bitstream/handle/2299/1918/901933.pdf?sequence=1) and plots empowerment landscape. 
    """  
    maze = MazeWorld(10,10)
    for i in range(6):
        maze.add_wall( (1, i), "N" )
    for i in range(2):
        maze.add_wall( (i+2, 5), "E")
        maze.add_wall( (i+2, 6), "E")
    maze.add_wall( (3, 6), "N")
    for i in range(2):
        maze.add_wall( (1, i+7), "N")
    for i in range(3):
        maze.add_wall( (5, i+2), "N")
    for i in range(2):
        maze.add_wall( (i+6, 5), "W")
    maze.add_wall( (6, 4), "N")
    maze.add_wall( (7, 4), "N")
    maze.add_wall( (8, 4), "W")
    maze.add_wall( (8, 3), "N")
    # compute the 5-step empowerment at each cell 
    E = maze.empowerment(n_step=5)
    # plot the maze world
    maze.plot(colorMap=E)
    plt.title('5-step empowerment')
    plt.show()

def example_2():
    """ builds grid world with doors and plots empowerment landscape """ 
    maze = MazeWorld(8,8)
    for i in range(maze.width):
        if i is not 6 : maze.add_wall([2, i], "N") 
    for i in range(maze.width):
        if i is not 2 : maze.add_wall([5, i], "N")
    n_step = 4
    E = maze.empowerment(n_step=n_step, n_samples=8000)
    maze.plot(colorMap=E)
    plt.title('%i-step empowerment' % n_step)
    plt.show()

def example_3():
    """ Runs empowerment maximising agent running in a chosen grid world """

    # maze = klyubin_world()
    maze = mazeworld.door_world()
    emptymaze = MazeWorld(maze.height, maze.width)
    # maze = mazeworld.tunnel_world()
    n_step = 3
    start = time.time()
    initpos = np.random.randint(maze.dims[0], size=2)
    initpos = [1,4]
    s =  maze._cell_to_index(initpos)
    T = emptymaze.compute_model()
    B = maze.compute_model()
    E = maze.empowerment(n_step = n_step).reshape(-1)
    n_s, n_a, _ = T.shape
    agent = EmpMaxAgent(alpha=0.1, gamma=0.9, T = T, n_step=n_step, n_samples=1000, det=1.)
    steps = int(10000) 
    visited = np.zeros(maze.dims)
    tau = np.zeros(steps)
    D_emp = np.zeros(steps)
    D_mod = n_s*n_a*np.ones(steps)
    for t in range(steps):
        # append data for plotting 
        tau[t] = agent.tau
        D_emp[t] = np.mean((E - agent.E)**2)
        D_mod[t] = D_mod[t] - np.sum(np.argmax(agent.T, axis=0) == np.argmax(B, axis=0))
        a = agent.act(s)
        pos = maze._index_to_cell(s)
        visited[pos[0],pos[1]] += 1
        s_ = maze.act(s,list(maze.actions.keys())[a])
        agent.update(s,a,s_)
        s = s_
    print("elapsed seconds: %0.3f" % (time.time() - start) )
    plt.figure(1)
    plt.title("value map")
    Vmap = np.max(agent.Q, axis=1).reshape(*maze.dims)
    maze.plot(colorMap= Vmap )
    plt.figure(2)
    plt.title("subjective empowerment")
    maze.plot(colorMap= agent.E.reshape(*maze.dims))
    plt.figure(3)
    plt.title("tau")
    plt.plot(tau)
    plt.figure(4)
    plt.scatter(agent.E, visited.reshape(n_s))
    plt.xlabel('true empowerment')
    plt.ylabel('visit frequency')
    plt.figure(5)
    plt.title("visited")
    maze.plot(colorMap=visited.reshape(*maze.dims))
    fig, ax1 = plt.subplots()
    red = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('MSE of empowerment map', color=red)
    ax1.plot(D_emp, color=red)
    ax1.tick_params(axis='y', labelcolor=red)
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Model disagreement', color='tab:blue')  
    ax2.plot(D_mod, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.show()

if __name__ == "__main__":
    ## uncomment below to see examples 
    example_1()
    # example_2()
    # example_3()