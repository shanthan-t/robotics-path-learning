import numpy as np
import matplotlib.pyplot as plt
GRID_SIZE = 50
START = (5, 5)
END = (45, 45)
OBSTACLES = [
    (15, 15, 5),
    (30, 25, 7),
    (10, 40, 6),
    (40, 10, 4)
]

def plot_environment(ax):
    """
    Plots the grid, start/end points, and obstacles on a given Axes object.
    """
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 5))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 5))
    ax.grid(True)
    ax.set_title("Robotics Path Planning Environment")
    ax.plot(START[0], START[1], 'go', markersize=10, label='Start')
    ax.plot(END[0], END[1], 'ro', markersize=10, label='End')
    for x, y, r in OBSTACLES:
        circle = plt.Circle((x, y), r, color='gray', alpha=0.5)
        ax.add_artist(circle)

POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1
PATH_LENGTH = 100 

MOVEMENTS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

def create_individual(path_length):
    """
    Creates a random path (individual) of a given length.
    A path is a list of movement vectors.
    """
    return [MOVEMENTS[np.random.randint(len(MOVEMENTS))] for _ in range(path_length)]

def create_population(size, path_length):
    """
    Creates a population of random paths.
    """
    return [create_individual(path_length) for _ in range(size)]

def get_path_coordinates(individual):
    """
    Converts a path of movements into absolute coordinates.
    """
    path_coords = [START]
    current_pos = START
    for move in individual:
        next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
        path_coords.append(next_pos)
        current_pos = next_pos
    return path_coords

def is_collision(x, y, obstacles):
    """
    Checks if a point (x, y) is inside any obstacle.
    """
    for obs_x, obs_y, r in obstacles:
        distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
        if distance <= r:
            return True
    return False

def calculate_fitness(individual):
    """
    Calculates the fitness of a path based on distance, collisions, and length.
    The goal is to MINIMIZE this score (lower is better).
    """
    path_coords = get_path_coordinates(individual)
    last_point = path_coords[-1]
    distance_to_end = np.sqrt((last_point[0] - END[0])**2 + (last_point[1] - END[1])**2)
    collision_penalty = 0
    for x, y in path_coords:
        if is_collision(x, y, OBSTACLES):
            collision_penalty = 500  # A large penalty for any collision
            break
            
    path_length_penalty = 0
    boundary_penalty = 0
    for x, y in path_coords:
        path_length_penalty += 1 
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            boundary_penalty = 500 
            break
    fitness = distance_to_end + collision_penalty + path_length_penalty + boundary_penalty
    return fitness

def selection(population, fitness_scores):
    """
    Selects parents using tournament selection.
    """
    pop_and_fitness = list(zip(population, fitness_scores))
    k = 5 
    parents = []
    for _ in range(len(population)):
        tournament = np.random.choice(pop_and_fitness, size=k, replace=False)
        best_in_tournament = min(tournament, key=lambda x: x[1])
        parents.append(best_in_tournament[0])   
    return parents

def crossover(parent1, parent2):
    """
    Performs single-point crossover between two parents to create offspring.
    """
    crossover_point = np.random.randint(1, len(parent1))
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2

def mutate(individual, mutation_rate):
    """
    Mutates an individual by randomly changing some movements.
    """
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = MOVEMENTS[np.random.randint(len(MOVEMENTS))] 
    return mutated_individual

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_environment(ax)
    population = create_population(POPULATION_SIZE, PATH_LENGTH)
    best_path = None
    best_fitness = float('inf')
    fitness_history = []
    
    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        min_fitness = min(fitness_scores)
        min_fitness_idx = fitness_scores.index(min_fitness)
        
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_path = population[min_fitness_idx]
        fitness_history.append(best_fitness)
        next_generation = []
        
        for _ in range(POPULATION_SIZE // 2):
            # Selection
            parent1 = selection(population, fitness_scores)[0]
            parent2 = selection(population, fitness_scores)[0]
            
            # Crossover
            offspring1, offspring2 = crossover(parent1, parent2)
            
            # Mutation
            offspring1 = mutate(offspring1, MUTATION_RATE)
            offspring2 = mutate(offspring2, MUTATION_RATE)
            
            next_generation.extend([offspring1, offspring2])    
        population = next_generation
        
        if generation % 20 == 0:
            print(f"Generation {generation+1}/{GENERATIONS} | Best Fitness: {best_fitness:.2f}")
    print("\nGenetic Algorithm Complete.")
    print(f"Final Best Fitness: {best_fitness:.2f}")
    
    ax.clear()
    plot_environment(ax)
    
    final_path_coords = get_path_coordinates(best_path)
    x_coords, y_coords = zip(*final_path_coords)
    ax.plot(x_coords, y_coords, 'b-', label='Optimal Path')
    ax.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history)
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score (lower is better)")
    plt.grid(True)
    plt.show()