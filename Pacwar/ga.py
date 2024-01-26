import numpy as np
import random
import _PyPacwar

def generate_individual():
    U = np.random.choice([0, 1, 2, 3], 4)
    W = np.random.choice([0, 1, 2, 3], 3)
    X = np.random.choice([0, 1, 2, 3], 3)
    V = np.random.choice([0, 1, 2, 3], (3, 4))
    Y = np.random.choice([0, 1, 2, 3], (3, 4))
    Z = np.random.choice([0, 1, 2, 3], (3, 4))
    return (U, W, X, V, Y, Z)

def convert_to_pacwar_format(individual):
    flattened_code = []
    for part in individual:
        flattened_code.extend(np.ravel(part))
    return flattened_code

def random_opponent():
    return [random.choice([0, 1, 2, 3]) for _ in range(50)]

def fitness(individual, population):
    pacwar_code = convert_to_pacwar_format(individual)
    opponent = random_opponent() if random.random() < 0.5 else convert_to_pacwar_format(random.choice(population))
    (rounds, c1, c2) = _PyPacwar.battle(pacwar_code, opponent)
    return rounds + c1 - c2

def select(population, fitness_scores):
    # Implement your selection method here (e.g., roulette wheel, tournament)
    # This is a placeholder for the selection process
    return sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:len(population)//2]

def crossover(parent1, parent2):
    child1, child2 = parent1.copy(), parent2.copy()
    crossover_point = random.randint(1, len(parent1)-2)
    child1[crossover_point:], child2[crossover_point:] = parent2[crossover_point:], parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for i, _ in enumerate(individual):
        if random.random() < mutation_rate:
            individual[i] = np.random.choice([0, 1, 2, 3], len(individual[i]))
    return individual

def tournament_selection(population, fitness_scores, tournament_size=5):
    selected = []
    for _ in range(len(population)):
        # Randomly select tournament_size individuals
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        # Select the best individual from the tournament
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

def genetic_algorithm(population_size=100, generations=50):
    population = [generate_individual() for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = [fitness(ind, population) for ind in population]
        print(f"Generation {gen+1}/{generations} - Best Fitness: {max(fitness_scores)}")

        # Use tournament selection
        selected = tournament_selection(population, fitness_scores)

        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.choices(selected, k=2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])

        population = next_generation

    best_individual = max(population, key=lambda ind: fitness(ind, population))
    return best_individual


best_params = genetic_algorithm()
print("Best Parameters:", best_params)
