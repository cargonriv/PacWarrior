# import numpy as np
import os
import _PyPacwar
import random
from pathlib import Path

NUM_RANDOMS = 2
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def save_best_individual(gene, directory=results_dir):
    with open(Path(directory) / 'best_individual.txt', 'w') as file:
        file.write(gene)


def load_best_individual(filename='best_individual.txt', directory=results_dir):
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


def fitness(warrior, best_performing_gene):
    ones = [1] * 50
    threes = [3] * 50 #TODO: put random
    opponents = [ones, threes, best_performing_gene]

    def physical(opponent):
        rounds, c1, c2 = _PyPacwar.battle(list(map(int, warrior)), list(map(int, opponent)))
    #     return rounds + c1 - c2  # Example fitness calculation
        #c1: remaining warrior mites, c2 = remaining opponent mites, round = number of rounds passed
    # return physical(warrior, ones) + physical(warrior, threes)
        # score = score_calc_searching(rounds, c1, c2)
        score = score_calc_efficiency(rounds, c1, c2)
        return score
        # Score combining rounds and remaining mites
        # return max(0, rounds + c1 - c2)
    # score_ones = physical(ones)
    # score_threes = physical(threes)

    scores_random = 0
    #TODO: figure out a good way to discourage bad performance (attempted in `score_calc`)
    # for i in range(NUM_RANDOMS): #measure of performance against random genes, weight
    #     scores_random += physical(generate_individual()) / NUM_RANDOMS * 1.0

    # Weighted average to balance performance
    # if total_score == 0:
    #     return 0
    # weight_threes = score_threes / total_score
    # weight_ones = score_ones / total_score
    # return weight_ones * score_ones + weight_threes * score_threes
    # total_score = score_ones + score_threes + scores_random

    # return total_score, score_ones, score_threes
    # total_score = sum([physical(opponent) for opponent in opponents])
    scores = [physical(opponent) for opponent in opponents]
    return sum(scores), scores[0], scores[1], scores[2]

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


def score_calc_efficiency(rounds, warrior, opp):
    base_score = 0
    if opp == 0:  # Victory
        base_score = 170  # Maximum score for defeating the opponent
        # Additional rewards based on efficiency (quick victories)
        if rounds < 100:
            base_score += 30
        elif rounds < 200:
            base_score += 25
        elif rounds < 300:
            base_score += 20
        elif rounds < 500:
            base_score += 15
    else:  # Battle wasn't won
        ratio = warrior / (opp + 0.1)  # Added a small value to prevent division by zero
        if ratio > 1:  # Warrior has more remaining units but didn't win
            base_score = ratio * 10  # Reward based on survival ratio
        else:
            # Penalty increases as the ratio decreases
            base_score = -20 * (1 - ratio)  # Penalty for losing, scaled by survival ratio

    # Encourage survival by adding a component based on the remaining number of warriors
    survival_bonus = warrior * 0.1
    return base_score + survival_bonus


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


def mutate(individual, mutation_rate):  # , mutation_rate=0.1):
    individual = list(individual)
    for i, _ in enumerate(individual):
        if random.random() < mutation_rate:
            individual[i] = str(random.choice([0, 1, 2, 3]))
    return ''.join(individual)


def tournament_selection(population, fitness_scores, tournament_size=3):
    selected_parents = []
    for _ in range(len(population) // tournament_size):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected_parents.append(winner)
    return selected_parents


def incorporate_elitism(population, fitness_scores, elite_size=2):
    sorted_by_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elites = [individual for individual, score in sorted_by_fitness[:elite_size]]
    return elites


def reproduce(selected, mutation_rate, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected, 2)
        child1, child2 = crossover(parent1, parent2)  # Assuming crossover is defined elsewhere
        new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])  # Assuming mutate is defined elsewhere
    return new_population[:population_size]


def test_ones_threes(warrior): 
    '''
    For testing against ones and threes
    '''
    ones = [1] * 50
    threes = [3] * 50
    rounds_ones, c1_ones, c2_ones = battle(warrior, ones)
    rounds_threes, c1_threes, c2_threes = battle(warrior, threes)
    print(f"Warrior vs All-Ones: Rounds = {rounds_ones}, Warriors Remaining = {c1_ones}, All-Ones Remaining = {c2_ones}")
    print(f"Warrior vs All-Threes: Rounds = {rounds_threes}, Warriors Remaining = {c1_threes}, All-Threes Remaining = {c2_threes}")

def test_random(warrior, num_opps, printV=False):
    '''
    Tests against num_opps number of random warriors. Returns raw score sum from the battles. 
    '''
    score = 0
    for trial in range(num_opps):
        new_gene = generate_individual()
        # print(warrior)
        warrior_lst = list(map(int, warrior))
        new_gene_lst = list(map(int, new_gene)) 
        rounds, c1, c2 = _PyPacwar.battle(warrior_lst, new_gene_lst)
        if printV:
            print(f"rounds: {rounds}, warrior count: {c1}, opp count: {c2}")
        score += score_calc_no_zero(rounds, c1, c2)

    return score

def test_100_random(warrior):
    '''
    Tests with 100 random warriors, returns average score
    '''
    return test_random(warrior, 100) / 100 

def test_lots_random(warrior):
    '''
    Tests with 100000 random warriors, returns average score
    '''
    return test_random(warrior, 100000) / 100000 



# Initialize storage for fitness scores and populations
fitness_scores_over_generations = []
populations_over_generations = []
def genetic_algorithm(population_size=50, generations=50, mutation_rate=0.02, crossover_rate=0.6, saved_individual=True, random=True):
    best_individual = [0]*50
    default_pop = [generate_individual() for _ in range(population_size - 1)]

    if not random:
        default_pop = [genetic_algorithm(population_size=50, generations=50, saved_individual=True, random=True) for _ in range(population_size - 1)]
        print(f"good genes: {default_pop}")
    if saved_individual:
        best_individual = load_best_individual()
        population = [best_individual] + default_pop
        best_fitness = fitness(best_individual, best_individual)[0]
    else:
        population = default_pop + [generate_individual()]
        best_fitness = None

    for gen in range(generations):
        fitness_scores = [fitness(ind, best_individual)[0] for ind in population]
        # if best_fitness is None or max(fitness_scores) > best_fitness:
        #     best_fitness = max(fitness_scores)
        #     best_individual = population[fitness_scores.index(best_fitness)]
        #     save_best_individual(best_individual)
        #     stagnation_counter = 0
        # else:
        #     print(f"stagnation #{stagnation_counter+1} on generation #{gen+1}")
        #     print(f"{len(fitness_scores)} max(fitness_scores):", max(fitness_scores), "; best_fitness:", best_fitness)
        #     stagnation_counter += 1

        # if stagnation_counter > 3:
        #     mutation_rate *= 1.5
        #     stagnation_counter = 0

        selected = tournament_selection(population, fitness_scores)
        population = reproduce(selected, mutation_rate, population_size)

        current_best_fitness = max(fitness_scores)
        current_best_index = fitness_scores.index(current_best_fitness)
        current_best_individual = population[current_best_index]

        # print(fitness(current_best_individual, best_individual), fitness(best_individual, best_individual))
        if not best_individual or fitness(current_best_individual, best_individual) > fitness(best_individual, best_individual):
            best_individual = current_best_individual
            best_fitness = current_best_fitness
            save_best_individual(best_individual)

        next_generation = []

        # fitness_scores_threes = [fitness(ind, best_individual)[2] for ind in population]
        # fitness_score_ones = [fitness(ind, best_individual)[1] for ind in population]
        fitness_scores_over_generations.append(max(fitness_scores))
        populations_over_generations.append(population.copy())

        # print(f"Generation {gen+1}/{generations} - Best Fitness from this Gen: {current_best_fitness}")
        print(f"Generation {gen+1}/{generations} - Best overall fitness: {best_fitness}")
        # print(f"Generation {gen+1}/{generations} - Best Fitness: {max(fitness_scores)}")
        # # print(len(set(population))) #number of unique genes in the population
        # #TODO: measure population change from generation to generation

        # # Tournament selection
        # # selected = [max(random.sample(list(zip(population, fitness_scores)), 20), key=lambda x: x[1])[0] for _ in population]
        # selected = [x[0] for x in sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:2]]
        # # print(len(set(selected))) #number of unique top selected genes
        # #TODO: store fitness data
        # next_generation = []
        # while len(next_generation) < population_size:
        #     next_generation.extend(find_next(selected))
        # population = next_generation
    # best_individual = max(population, key=lambda ind: fitness(ind)[0])
    # save_best_individual(best_individual)

    return best_individual

# def find_next_really_random(selected):
#     parent1, parent2 = random.choices(selected, k=2)
#     child1, child2 = crossover(parent1, parent2)
#     return [mutate(child1), mutate(child2), child1, child2]

def find_next(selected):
    parent1, parent2 = selected[0], selected[1]
    child1, child2 = crossover(parent1, parent2)
    return [child1, child2, mutate(parent1), mutate(parent2)]

# pacwarrior_gene = genetic_algorithm(population_size=50, generations=50, saved_individual=False, random=False)

# print("Optimized PacWarrior Gene:", pacwarrior_gene)

# best_i = load_best_individual()
# print(best_i)
# next_i = '13000000111122203320213212233013213213313113102311'
# print(battle(best_i, next_i))
# print(test_ones_threes(best_i))

# print(test_random(best_i, 20, True))

# print(test_lots_random(best_i))
# for i in range(20):
#     print(f"score {i}: {test_100_random(best_i)}")

def run_sub_algorithm(x):
    good_genes = []
    for i in range(x):
        print("iteration #", i+1)
        good_gene = genetic_algorithm(random=False)
        good_genes.append(good_gene)
        print(good_genes)
    # Run a tournament among the good genes
    best_gene = robust_tournament(good_genes)
    return best_gene

def robust_tournament(genes):
    """
    Conduct a round-robin tournament where each gene battles every other gene.
    Scores from each battle are accumulated to find the gene with the best overall performance.
    """
    scores = {gene: 0 for gene in genes}  # Initialize scores for each gene
    
    # Conduct battles
    for i, gene1 in enumerate(genes):
        for gene2 in genes[i+1:]:  # Avoid repeating battles and battling itself
            score1 = fitness(gene1, gene2)[0]
            score2 = fitness(gene2, gene1)[0]
            scores[gene1] += score1
            scores[gene2] += score2

    # Find the gene with the highest score
    best_gene = max(scores, key=scores.get)
    return best_gene

def test_against_pacwar_ii_gene(candidate_gene):
    pacwar_ii_gene = "01320000133122023333133323222133133222133110301310"
    # pacwar_ii_gene = "01300030103021123333233323223030222032323310200310"
    rounds, c1, c2 = _PyPacwar.battle(list(map(int, candidate_gene)), list(map(int, pacwar_ii_gene)))
    print(f"Test Against Pacwar II Gene: Rounds = {rounds}, Candidate Remaining = {c1}, Pacwar II Remaining = {c2}")

# Example usage
x = 50  # Adjust based on desired execution time and experimentation
best_gene = run_sub_algorithm(x)
print(f"Best Gene: {best_gene}")
test_against_pacwar_ii_gene(best_gene)
