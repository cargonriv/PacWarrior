# import numpy as np
import _PyPacwar
import random


def generate_individual():
    return ''.join(str(random.choice([0, 1, 2, 3])) for _ in range(50))


def battle(gene1, gene2):
    rounds, c1, c2 = _PyPacwar.battle(list(map(int, gene1)), list(map(int, gene2)))
    return rounds, c1, c2


def fitness(champion):
    ones = [1] * 50
    threes = [3] * 50

    def physical(opponent):
        rounds, c1, c2 = _PyPacwar.battle(list(map(int, champion)), list(map(int, opponent)))
    #     return rounds + c1 - c2  # Example fitness calculation
        #c1: remaining warrior mites, c2 = remaining opponent mites, round = number of rounds passed
    # return physical(champion, ones) + physical(champion, threes)
        score = score_calc_searching(rounds, c1, c2)
        return score
        # return max(0, rounds + c1 - c2)  # Score combining rounds and remaining mites
    score_ones = physical(ones)
    score_threes = physical(threes)

    # Weighted average to balance performance
    #TODO: figure out a good way to discourage bad performance (done in score_calc_searching)
    total_score = score_ones + score_threes

    # if total_score == 0:
    #     return 0
    # weight_threes = score_threes / total_score
    # weight_ones = score_ones / total_score

    # return weight_ones * score_ones + weight_threes * score_threes
    return total_score

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


def score_calc_searching(round, warrior, opp):
    """
    rounds done, warrior mites remaining, opps remaining
    good for getting other opponents defeated
    at worst, we will not defeat the opp in 500 rounds, leaving some ratio of Warrior : opp. 
    higher ratio is better, max ratio is 9*19 (dimension of grid) = 171 - 1 = 170 : 1
    at best, we defeat the opponent. score is based on the canvas thing (added 170)

    Max score is 190
    TODO: if we get stuck in 0, then may want to encode remaining opp mites in score
    """ 
    #max rounds: 500
    if (opp == 0):
        if (round < 100):
            return 170 + 20
        if (round < 200):
            return 170 + 19
        if (round < 300):
            return 170 + 18
        if (round < 501):
            return 170 + 17
    ratio = warrior / opp * 1.0 #ratio of warrior mites remaining to opp mites remaining after 500 rounds

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


def mutate(individual, mutation_rate=0.02):
    individual = list(individual)
    for i, _ in enumerate(individual):
        if random.random() < mutation_rate:
            individual[i] = str(random.choice([0, 1, 2, 3]))
    return ''.join(individual)


def test(champion):
    ones = [1] * 50
    threes = [3] * 50
    rounds_ones, c1_ones, c2_ones = battle(champion, ones)
    rounds_threes, c1_threes, c2_threes = battle(champion, threes)
    print(f"Champs vs All-Ones: Rounds = {rounds_ones}, Champs Remaining = {c1_ones}, All-Ones Remaining = {c2_ones}")
    print(f"Champs vs All-Threes: Rounds = {rounds_threes}, Champs Remaining = {c1_threes}, All-Threes Remaining = {c2_threes}")


def genetic_algorithm(population_size=100, generations=100):
    population = [generate_individual() for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = [fitness(ind) for ind in population]
        print(f"Generation {gen+1}/{generations} - Best Fitness: {max(fitness_scores)}")

        # Tournament selection
        selected = [max(random.sample(list(zip(population, fitness_scores)), 5), key=lambda x: x[1])[0] for _ in population]

        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.choices(selected, k=2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])

        population = next_generation

    best_individual = max(population, key=lambda ind: fitness(ind))
    return best_individual


neurostrategist_gene = genetic_algorithm()

print("Optimized Champ Gene:", neurostrategist_gene)

test(neurostrategist_gene)
