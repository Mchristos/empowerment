# Empowerment

## What is Empowerment

Reinforcement learning is a powerful tool for teaching agents complex skills based on experience. However, it relies upon a supervisor or "critic" to dole out rewards, defining a norm for "good" behaviour externally. In the biological world, such supervisors rarely exist, and creatures must learn about their environments independently and autonomously.

In the absence of specific rewards provided by external entities, autonomous agents should be self-motivated, and driven by behaving in such a way that makes them more prepared for future tasks or challenges. **Empowerment** has been proposed as an **intrinsic motivation** for autonomous behaviour [in this paper](https://uhra.herts.ac.uk/bitstream/handle/2299/1918/901933.pdf?sequence=1) [1]. I've also written about empowerment [in this article](https://towardsdatascience.com/empowerment-as-intrinsic-motivation-b84af36d5616). In other words, empowerment can be used as a motivator for agents when there are no obvious rewards. That's because empowered states tend to be "interesting", "manipulatable" states, where ones actions potentially have a lot of effect. 

Empowerment measures how much power an agent has to influence the state of its environment. Crucially, though, it measures only that change which is detectable via the sensors. In other words - how much influence do I have over my future sensory states? To quantify this precisely, we consider the information-theoretic channel describing how actions influence subsequent sensory states. The concept can easily be generalized to consider sequences of n actions: how much can I inluence the state of the environment over n steps? Below we illustrate these information-theoretic channels defining empowerment. 

<img width="641" alt="emp_network" src="https://user-images.githubusercontent.com/13951953/44619336-b15eef00-a87c-11e8-9fea-6eb8c564fbb7.png">

Rt, St, and At represent the state of the environment, the sensory state and the action taken at time t, respectively. 

## What is this repo?

This repository allows you to calculate empowerment and experiment with it in various settings. As mentioned in the previous section, empowerment measures the channel capacity of the information-theoretic channel describing how actions determine future states.

- mazeworld.py provides a class MazeWorld which allows you to create arbitrary grid worlds with walls like in the examples beloew, and compute the empowerment of states 
- empowerment.py is a module allowing you to compute empowerment in arbitrary environments described by a probabilistic transition rule p(s'|s,a) - probability of landing in state s' given you did action a in state s.
- info_theory.py is a module containing various functions for computing information-theoretic quantities such as entropy, conditional entropy, mutual information, and channel capacity. It includes an implementation of the blahut-arimoto algorithm for channel capacity. This is used to compute the empowerment in non-deterministic environments. 

## Example Usage 

### Klyubin's Maze World 
In this example we reproduce the grid world presented in the original Klyubin paper on empowerment [1]. 

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
    E = maze.empowerment(n_step=5)
    # plot the maze world
    maze.plot(colorMap=E)
    plt.title('5-step empowerment')
    plt.show()

<img width="500" alt="" src="https://user-images.githubusercontent.com/13951953/44622686-80e77700-a8b5-11e8-9386-738a9fdbb56b.png">

### Doorways
Here is another simple example of a small grid world with two doorways. The doorways are highly empowered in 4 steps, since more states are reachable within four steps from doorways. 

    maze = MazeWorld(8,8)
    for i in range(maze.width):
        if i is not 6 : maze.add_wall([2, i], "N") 
    for i in range(maze.width):
        if i is not 2 : maze.add_wall([5, i], "N")
    n_step = 4
    E = maze.empowerment(n_step=n_step)
    maze.plot(colorMap=E)
    plt.title('%i-step empowerment' % n_step)
    plt.show()
    
<img width="500" alt="" src="https://user-images.githubusercontent.com/13951953/44622721-40d4c400-a8b6-11e8-85e9-ee503e0319c8.png">

[1] Klyubin, A.S., Polani, D. and Nehaniv, C.L., 2005, September. All else being equal be empowered. In European Conference on Artificial Life (pp. 744-753). Springer, Berlin, Heidelberg.
