# Empowerment

## What is Empowerment

Reinforcement learning is a powerful tool for teaching agents complex skills based on experience. However, it relies upon a supervisor or "critic" to dole out rewards, defining a norm for "good" behaviour externally. In the biological world, such supervisors rarely exist, and creatures must learn about their environments independently and autonomously.

In the absence of specific rewards provided by external entities, autonomous agents should be self-motivated, and driven by behaving in such a way that makes them more prepared for future tasks or challenges. **Empowerment** has been proposed as an **intrinsic motivation** for autonomous behaviour [in this paper](https://uhra.herts.ac.uk/bitstream/handle/2299/1918/901933.pdf?sequence=1) [1]. I've also written about empowerment [in this article](https://towardsdatascience.com/empowerment-as-intrinsic-motivation-b84af36d5616). 

Empowerment measures how much power an agent has to influence the state of its environment. 

## What is this repo?

This repository allows you to calculate empowerment and experiment with it in various settings. As mentioned in the previous section, empowerment measures the channel capacity of the information-theoretic channel describing how actions determine future states.

## Example: Klyubin's Maze World 
In this section we reproduce the grid world presented in the original Klyubin paper on empowerment [1]. 

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

<img width="641" alt="5stepempowerment" src="https://user-images.githubusercontent.com/13951953/44586448-9d958900-a7a7-11e8-97dd-51ff4fc223d0.png">


[1] Klyubin, A.S., Polani, D. and Nehaniv, C.L., 2005, September. All else being equal be empowered. In European Conference on Artificial Life (pp. 744-753). Springer, Berlin, Heidelberg.
