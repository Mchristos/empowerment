from mazeworld import MazeWorld, empowerment
import matplotlib.pyplot as plt

# build klyubin maze world 
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
E = empowerment(maze, n_step=5, n_samples=1000)
# plot the maze world
maze.plot(colorMap=E)
plt.title('5-step empowerment')
plt.show()