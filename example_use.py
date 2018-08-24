from mazeworld import MazeWorld, empowerment
import matplotlib.pyplot as plt 

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

E = empowerment(maze, n_step=3, n_samples=1000, det=0.8)
maze.plot(colorMap=E)
plt.show()