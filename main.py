import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
GRID_SIZE = 50
START = (5, 5)
END = (45, 45)
OBSTACLES = [
    (15, 15, 5),
    (30, 25, 7),
    (10, 40, 6),
    (40, 10, 4)
]
DYNAMIC_OBSTACLE = {
    'start_pos': (0, 25),
    'end_pos': (50, 25),
    'radius': 4
}

def plot_environment(ax):
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
    ax.plot(DYNAMIC_OBSTACLE['start_pos'][0], DYNAMIC_OBSTACLE['start_pos'][1], 'y+', markersize=10, label='Dynamic Obs. Start')
    ax.plot(DYNAMIC_OBSTACLE['end_pos'][0], DYNAMIC_OBSTACLE['end_pos'][1], 'y_', markersize=10, label='Dynamic Obs. End')

POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1
ELITISM_RATE = 0.1
PATH_LENGTH = 100
MOVEMENTS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

def create_individual(path_length):
    return [MOVEMENTS[np.random.randint(len(MOVEMENTS))] for _ in range(path_length)]

def create_population(size, path_length):
    return [create_individual(path_length) for _ in range(size)]

def get_path_coordinates(individual):
    path_coords = [START]
    current_pos = START
    for move in individual:
        next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
        path_coords.append(next_pos)
        current_pos = next_pos
    return path_coords

def is_collision(x, y, obstacles, t=None):
    for obs_x, obs_y, r in obstacles:
        distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
        if distance <= r:
            return True
            
    if t is not None:
        dynamic_x = DYNAMIC_OBSTACLE['start_pos'][0] + (DYNAMIC_OBSTACLE['end_pos'][0] - DYNAMIC_OBSTACLE['start_pos'][0]) * (t / PATH_LENGTH)
        dynamic_y = DYNAMIC_OBSTACLE['start_pos'][1] + (DYNAMIC_OBSTACLE['end_pos'][1] - DYNAMIC_OBSTACLE['start_pos'][1]) * (t / PATH_LENGTH)
        dynamic_r = DYNAMIC_OBSTACLE['radius']
        distance_dynamic = np.sqrt((x - dynamic_x)**2 + (y - dynamic_y)**2)
        if distance_dynamic <= dynamic_r:
            return True
    return False

def calculate_fitness(individual):
    path_coords = get_path_coordinates(individual)
    last_point = path_coords[-1]
    distance_to_end = np.sqrt((last_point[0] - END[0])**2 + (last_point[1] - END[1])**2)
    collision_penalty = 0
    boundary_penalty = 0
    path_smoothness_penalty = 0
    
    for t, (x, y) in enumerate(path_coords):
        if is_collision(x, y, OBSTACLES, t):
            collision_penalty = 500
            break
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            boundary_penalty = 500
            break
            
    for i in range(len(individual) - 1):
        move1 = individual[i]
        move2 = individual[i+1]
        if move1 == (-move2[0], -move2[1]):
            path_smoothness_penalty += 10 
    path_length_penalty = len(path_coords)
    fitness = distance_to_end + collision_penalty + path_length_penalty + boundary_penalty + path_smoothness_penalty
    return fitness

def selection(population, fitness_scores):
    pop_and_fitness = list(zip(population, fitness_scores))
    parents = []
    tournament_size = 5
    for _ in range(len(population)):
        tournament = random.sample(pop_and_fitness, k=tournament_size)
        best_in_tournament = min(tournament, key=lambda x: x[1])
        parents.append(best_in_tournament[0])
    return parents

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2

def mutate(individual, mutation_rate):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = MOVEMENTS[np.random.randint(len(MOVEMENTS))]
    return mutated_individual

def run_ga(ax):
    global START, END
    population = create_population(POPULATION_SIZE, PATH_LENGTH)
    best_path = None
    best_fitness = float('inf')
    fitness_history = []
    num_elites = int(POPULATION_SIZE * ELITISM_RATE)
    
    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        pop_with_fitness = sorted(list(zip(population, fitness_scores)), key=lambda x: x[1])
        elites = [ind for ind, fit in pop_with_fitness[:num_elites]]
        min_fitness = pop_with_fitness[0][1]
        
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_path = pop_with_fitness[0][0]
        fitness_history.append(best_fitness)
        next_generation = elites[:] 
        parents = selection(population, fitness_scores)
        random.shuffle(parents)

        for i in range(0, POPULATION_SIZE - num_elites, 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, MUTATION_RATE)
            offspring2 = mutate(offspring2, MUTATION_RATE)
            next_generation.extend([offspring1, offspring2])
        population = next_generation
        
        if generation % 20 == 0:
            print(f"Generation {generation+1}/{GENERATIONS} | Best Fitness: {best_fitness:.2f}")

    print("\nGenetic Algorithm Complete.")
    print(f"Final Best Fitness: {best_fitness:.2f}")
    animate_solution(ax, best_path, fitness_history)

def animate_solution(ax, best_path, fitness_history):
    fig = ax.figure
    final_path_coords = get_path_coordinates(best_path)
    path_x, path_y = zip(*final_path_coords)
    robot_line, = ax.plot([], [], 'b-')
    robot_point, = ax.plot([], [], 'bo', markersize=8)
    dynamic_obs_point, = ax.plot([], [], 'yo', markersize=8)
    ax.clear()
    plot_environment(ax)
    ax.plot(path_x, path_y, 'b--', alpha=0.5)

    def update(frame):
        ax.set_title(f"Robot Path Animation | Step {frame}/{PATH_LENGTH}")
        robot_line.set_data(path_x[:frame+1], path_y[:frame+1])
        robot_point.set_data([path_x[frame]], [path_y[frame]])
        dynamic_x = DYNAMIC_OBSTACLE['start_pos'][0] + (DYNAMIC_OBSTACLE['end_pos'][0] - DYNAMIC_OBSTACLE['start_pos'][0]) * (frame / PATH_LENGTH)
        dynamic_y = DYNAMIC_OBSTACLE['start_pos'][1] + (DYNAMIC_OBSTACLE['end_pos'][1] - DYNAMIC_OBSTACLE['start_pos'][1]) * (frame / PATH_LENGTH)
        dynamic_obs_point.set_data([dynamic_x], [dynamic_y])
        
        return robot_line, robot_point, dynamic_obs_point

    ani = FuncAnimation(fig, update, frames=PATH_LENGTH, interval=100, blit=True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history)
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score (lower is better)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    click_count = 0
    def on_click(event):
        global click_count, START, END
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        ax.clear()
        plot_environment(ax)

        if click_count == 0:
            START = (x, y)
            ax.plot(START[0], START[1], 'go', markersize=10, label='Start')
            plt.title("Click to set End point")
        elif click_count == 1:
            END = (x, y)
            ax.plot(END[0], END[1], 'ro', markersize=10, label='End')
            plt.title("End point set. Running GA...")
        fig.canvas.draw_idle()
        click_count += 1
        if click_count == 2:
            plt.pause(1)
            run_ga(ax)
            click_count = 0 
    fig.canvas.mpl_connect('button_press_event', on_click)
    plot_environment(ax)
    plt.title("Click to set Start point")
    plt.show()