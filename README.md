Robotics Path Planning with a Genetic Algorithm

Overview
This project demonstrates the use of a genetic algorithm (GA) to solve a classic robotics problem: finding an optimal, collision-free path for a robot from a starting point to an endpoint in a static 2D environment with obstacles. The genetic algorithm, inspired by natural selection, "evolves" a population of potential paths over many generations to find a highly effective solution.

How the Genetic Algorithm Works
The core of this project lies in the application of the genetic algorithm to the path-planning problem. The key components are defined as follows:

Chromosome (Path): A single path is represented as a sequence of discrete movements (e.g., (1,0) for right, (-1,1) for up-left). A collection of these movements forms a single "individual" or "chromosome."

Population: A set of many different, randomly generated paths.

Fitness Function: This is a crucial metric that evaluates how "good" a path is. It's designed to minimize the final score, where a lower score indicates a more optimal path. The fitness score is a sum of three criteria:

Distance to End: The Euclidean distance from the path's final point to the target.

Collision Penalty: A large penalty is applied if any part of the path collides with an obstacle.

Path Length: A small penalty is applied for each step to encourage shorter, more efficient paths.

Selection: Parents for the next generation are chosen based on their fitness. This project uses a tournament selection method, where a small group of individuals is randomly chosen and the fittest among them is selected to be a parent.

Crossover (Recombination): Two parent paths are combined at a random point to create two new offspring, inheriting traits from both parents.

Mutation: Small, random changes are introduced to the offspring paths. This is essential for exploring new solutions and preventing the algorithm from getting stuck in local minima.

How to Run
To run this project, you need to have Python installed on your system.

Prerequisites
Make sure you have numpy and matplotlib installed. If you created a virtual environment, activate it first.

pip install numpy matplotlib

Execution
Simply run the main Python script from your terminal:

python main.py

The script will print the fitness evolution over generations and display two plots:

The final, optimized path plotted on the environment.

A graph showing the improvement of the best fitness score over generations.

Potential Future Work
This project can be extended in several ways to increase its complexity and applicability:

Dynamic Environments: Modify the algorithm to handle moving obstacles in real-time.

Multiple Robots: Plan paths for multiple robots, avoiding collisions with each other.

Different Fitness Criteria: Add criteria such as energy consumption or path smoothness to the fitness function.

Visualization Improvements: Create an animated visualization of the population's evolution over generations to better illustrate the algorithm's progress.