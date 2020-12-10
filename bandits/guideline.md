# Global Objective
During this practical session you will implement several Bandit algorithms and the needed framework to compare them on artificial data.

# Source Code
## Requisites
* To run online
	* copy and run `tutorial.ipynb` file on google-lab
* To run on you own computer
	* install	
		* Python3
		* `numpy` and `scipy` for tensor manipulation
		* `matplotlib` for graphics
		* Jupyter-Notebook for the tutorial
		* `docopt` for options of `play_games.py`
	* clone the git-hub repository and add a directory `data` 

```bash
git clone https://github.com/gaudel/recommender_system.git
mkdir recommender_system/bandits/data
```

## Initial
The initial version of the source code has been split in several files:

* *player.py*: some bandit algorithms implemented as objects with three functions
	* `choose_next_arm() -> unsigned int`: choose the next arm to pull
	* `update(unsigned int: arm, float: reward) -> void`: update stored informations given that the arm `arm` was pulled 	* `restart() -> void`: erase player's memory
* *arm.py*: some standard arms implemented as objects with two functions
	* `draw() -> float`: return a random value given the probability distribution corresponding to the arm
	* `mean() -> float`: return the expected reward corresponding to the probability distribution of the arm
* *play_games.py*: a script to run experiments and store the results in a file.
	* example: ```python3 play_games.py 200 10 --Random --Ber 0.4 0.2 0.8```
	* more info: ```python3 play_games.py -h```
* *exp.py*, *tools.py*: a few useful functions to run experiments and plot results.

# Mandatory Job During the Practical Session
In almost chronological order:

## Phase 0: Discover the APIs
* Read *tutorial.ipynb*, fill the missing code lines, and run.
* Take a look at already implemented players and arms.

## Phase 1: UCB1
* Implement the UCB1 strategy : $a_t = argmax_a~\hat\mu_t + \sqrt{\frac{2\log t}{T_a(t)}}$
* Compare $\varepsilon_n$-greedy, UCB1, and Thompson Sampling behavior  
	* focus on the cumulative regret curve (given time-step)
	* fix the environment to two arms {Bernoulli(0.2), Bernoulli(0.5)}, with horizon 300
	* optimise the $\varepsilon_n$-greedy parameter and the UCB parameter 
	* plot results of the selected parameters
* Do the same job against the set of two arms {Bernoulli(0.2), Bernoulli(0.9)}, with horizon 300
		* are the best parameters the same ? How behave the parameters selected during previous optimization
* Idem with horizon 10,000

## Bonus
* Implement Bernoulli arms.
* Compare Algorithms with more divers environments (more arms, Bernoulli arms...).
* Look for the worst set of 10 arms for Explore Then Commit (with horizon 300). How behave UCB1 and $\varepsilon_n$-greedy against that setting ?
* Look for the worst set of 10 arms for UCB / $\varepsilon_n$-greedy.
