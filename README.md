**Robotics Path Planning with a Genetic Algorithm**

**Overview**

This project uses a genetic algorithm to find an optimal path for a robot. The path must navigate around obstacles to get from a start point to an end point.

**How It Works (In a Nutshell)**: 
The program **"evolves"** a solution over many generations, much like a natural process.

**Starts with Random Paths:** The algorithm begins with a "population" of random, unoptimized paths.

**Scores Each Path:** Each path is given a "fitness score." A lower score means a better path. The score is based on how short the path is and how well it avoids hitting obstacles.

**Finds the Best Paths:** The paths with the best scores are selected to become "parents" for the next generation.

**Creates New Paths:** New paths are created by combining segments from the parent paths. Small, random changes are also introduced to ensure new solutions are explored.

**Repeats:** This process is repeated over hundreds of generations until a highly optimized, obstacle-free path is found.

**How to Run**
This project requires Python along with numpy and matplotlib.

**Install Libraries:**
pip install numpy matplotlib

**Run the Script:**
python main.py

**Interact:** A window will pop up. Click on the grid to set your start and end points, and the program will automatically find the best path.
