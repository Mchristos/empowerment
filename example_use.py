from mazeworld import MazeWorld, empowerment
import matplotlib.pyplot as plt

def klubin_example():      
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

def door_example():
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

if __name__ == "__main__":
    door_example()