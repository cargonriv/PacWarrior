# import numpy as np
import os
import _PyPacwar
import random
from pathlib import Path

NUM_RANDOMS = 2
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def save_best_individual(gene, directory):
    with open(Path(directory) / 'best_individual.txt', 'w') as file:
        file.write(gene)


def load_best_individual(filename='best_individual.txt', directory='results'):
    try:
        with open(Path(directory) / filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


def generate_individual():
    return ''.join(str(random.choice([0, 1, 2, 3])) for _ in range(50))


def battle(gene1, gene2):
    rounds, c1, c2 = _PyPacwar.battle(list(map(int, gene1)), list(map(int, gene2)))
    return rounds, c1, c2


def fitness(warrior):
    ones = [1] * 50
    threes = [3] * 50

    def physical(opponent):
        rounds, c1, c2 = _PyPacwar.battle(list(map(int, warrior)), list(map(int, opponent)))
    #     return rounds + c1 - c2  # Example fitness calculation
        #c1: remaining warrior mites, c2 = remaining opponent mites, round = number of rounds passed
    # return physical(warrior, ones) + physical(warrior, threes)
        score = score_calc_searching(rounds, c1, c2)
        return score
        # return max(0, rounds + c1 - c2)  # Score combining rounds and remaining mites
    score_ones = physical(ones)
    score_threes = physical(threes)

    scores_random = 0
    # for i in range(NUM_RANDOMS): #measure of performance against random genes, weight
    #     scores_random += physical(generate_individual()) / NUM_RANDOMS * 1.0

    # Weighted average to balance performance
    #TODO: figure out a good way to discourage bad performance (done in score_calc_searching)
    total_score = score_ones + score_threes + scores_random

    # if total_score == 0:
    #     return 0
    # weight_threes = score_threes / total_score
    # weight_ones = score_ones / total_score

    # return weight_ones * score_ones + weight_threes * score_threes
    return total_score, score_ones, score_threes

def score_calc(round, warrior, opp):
    #rounds done, warrior mites remaining, opps remaining
    #real one, based on canvas
    #max rounds: 500
    if (opp == 0):
        if (round < 100):
            return 20
        if (round < 200):
            return 19
        if (round < 300):
            return 18
        if (round < 501):
            return 17
    ratio = warrior / opp * 1.0 #ratio of warrior mites remaining to opp mites remaining after 500 rounds

    if (ratio > 10):
        return 13
    if (ratio > 3):
        return 12
    if (ratio > 1.5):
        return 11
    if (ratio < 1.5) and (ratio > 2 / 3.0):
        return 10
    return 0


def score_calc_searching(rounds, warrior, opp):
    """
    rounds done, warrior mites remaining, opps remaining
    good for getting other opponents defeated
    at worst, we will not defeat the opp in 500 rounds, leaving some ratio of Warrior : opp. 
    higher ratio is better, max ratio is 9*19 (dimension of grid) = 171 - 1 = 170 : 1
    at best, we defeat the opponent. score is based on the canvas thing (added 170)
    """ 
    #max rounds: 500
    if (opp == 0):
        if (rounds < 100):
            return 7 + 20
        if (rounds < 200):
            return 7 + 19
        if (rounds < 300):
            return 7 + 18
        if (rounds < 501):
            return 7 + 17
    ratio = warrior / opp * 1.0 #ratio of warrior mites remaining to opp mites remaining after 500 rounds
    if ratio == 0: #discourage losing by having a negative score based on remaining opponents 
        return -opp
    return ratio


def score_calc_no_zero(round, warrior, opp):
    #rounds done, warrior mites remaining, opps remaining
    #
    #max rounds: 500
    if (opp == 0):
        if (round < 100):
            return 20
        if (round < 200):
            return 19
        if (round < 300):
            return 18
        if (round < 501):
            return 17
    ratio = warrior / opp * 1.0 #ratio of warrior mites remaining to opp mites remaining after 500 rounds

    if (ratio > 10):
        return 13
    if (ratio > 3):
        return 12
    if (ratio > 1.5):
        return 11
    if (ratio < 1.5) and (ratio > 2 / 3.0):
        return 10
    return ratio


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate=0.1):
    individual = list(individual)
    for i, _ in enumerate(individual):
        if random.random() < mutation_rate:
            individual[i] = str(random.choice([0, 1, 2, 3]))
    return ''.join(individual)


def test(warrior):
    ones = [1] * 50
    threes = [3] * 50
    rounds_ones, c1_ones, c2_ones = battle(warrior, ones)
    rounds_threes, c1_threes, c2_threes = battle(warrior, threes)
    print(f"Warrior vs All-Ones: Rounds = {rounds_ones}, Warriors Remaining = {c1_ones}, All-Ones Remaining = {c2_ones}")
    print(f"Warrior vs All-Threes: Rounds = {rounds_threes}, Warriors Remaining = {c1_threes}, All-Threes Remaining = {c2_threes}")


# Initialize storage for fitness scores and populations
fitness_scores_over_generations = []
populations_over_generations = []
def genetic_algorithm(population_size=100, generations=100, saved_individual=True):
    if saved_individual:
        population = [load_best_individual()] + [generate_individual() for _ in range(population_size - 1)]
    else:
        population = [generate_individual() for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = [fitness(ind)[0] for ind in population]
        fitness_score_ones = [fitness(ind)[1] for ind in population]
        fitness_scores_threes = [fitness(ind)[2] for ind in population]
        fitness_scores_over_generations.append(max(fitness_scores))
        populations_over_generations.append(population.copy())
        print(f"Generation {gen+1}/{generations} - Best Fitness: {max(fitness_scores)}")
        # print(len(set(population))) #number of unique genes in the population
        #TODO: measure population change from generation to generation

        # Tournament selection
        # selected = [max(random.sample(list(zip(population, fitness_scores)), 20), key=lambda x: x[1])[0] for _ in population]
        selected = [x[0] for x in sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:2]]
        # print(len(set(selected))) #number of unique top selected genes
        #TODO: store fitness data 
        next_generation = []
        while len(next_generation) < population_size:
            next_generation.extend(find_next(selected))
        population = next_generation

    best_individual = max(population, key=lambda ind: fitness(ind))
    save_best_individual(best_individual, results_dir)
    return best_individual

# def find_next_really_random(selected):
#     parent1, parent2 = random.choices(selected, k=2)
#     child1, child2 = crossover(parent1, parent2)
#     return [mutate(child1), mutate(child2), child1, child2]

def find_next(selected):
    parent1, parent2 = selected[0], selected[1]
    child1, child2 = crossover(parent1, parent2)
    return [child1, child2, mutate(parent1), mutate(parent2)]

pacwarrior_gene = genetic_algorithm()

print("Optimized PacWarrior Gene:", pacwarrior_gene)

test(pacwarrior_gene)
