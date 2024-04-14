import math
import pathlib
import random
import numpy as np
import _PyPacwar  # type: ignore


def load_best_individual(filename='best_genes', directory='results'):
    try:
        with open(pathlib.Path(directory) / f'{filename}.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


def save_best_individual(gene, directory='results', filename='best_genes'):
    with open(pathlib.Path(directory) / f'{filename}.txt', 'a') as file:
        file.write(''.join(map(str, gene)) + '\n')


def generate_individual():
    return [random.choice(DIGITS) for _ in range(GENE_LENGTH)]


def compete(gene1, gene2):
    _, c1, c2 = _PyPacwar.battle(list(map(int, gene1)), list(map(int, gene2)))
    return (gene1, gene2) if c1 > c2 else (gene2, gene1)


def evaluate_gene(gene, population_size):
    # opponent_genes = [generate_individual() for _ in range(population_size)]
    opponent_genes = [generate_individual() for _ in range(6)]
    opponent_genes.append([1] * GENE_LENGTH)
    opponent_genes.append([3] * GENE_LENGTH)
    opponent_genes.append(list(map(int, list('01320000133122023333133323222133133222133110301310'.strip()))))
    opponent_genes.append(list(map(int, list('01300030103021123333233323223030222032323310200310'.strip()))))
    
    score = 0
    for opponent_gene in opponent_genes:
        rounds, c1, c2 = _PyPacwar.battle(
            list(map(int, gene)),
            list(map(int, opponent_gene))
        )
        score += c1 - c2 + max(0, 500 - rounds)
    return score / len(opponent_genes)


def mutate(gene, score, avg_previous_score):
    new_gene = gene[:]
    if (score - avg_previous_score) < 0:
        # Ensure mutation rate wont exceed 100%
        mutation_rate = min(BASE_RATE * 2, 1.0)
    else:
        # Dynamic mutation based on score
        mutation_rate = BASE_RATE / (1 + math.exp(-(score - avg_previous_score) / 100))
    # Ensure at least one mutation
    # mutation_points = max(1, int(mutation_rate * GENE_LENGTH))
    mutation_indices = random.sample(range(GENE_LENGTH), max(1, int(mutation_rate * GENE_LENGTH)))
    for i in mutation_indices:
        new_gene[i] = random.choice(DIGITS)
    return new_gene


def crossover(gene1, gene2):
    index = random.randint(1, GENE_LENGTH - 2)
    return gene1[:index] + gene2[index:], gene2[:index] + gene1[index:]


def select_parents(population, scores):
    tournament_size = 5
    selected_parents = []
    while len(selected_parents) < 2:
        contenders = random.sample(
            list(zip(population, scores)),
            min(tournament_size, len(population))
        )
        selected_parents.append(
            max(contenders, key=lambda contestant: contestant[1])[0]
        )

    return selected_parents


def create_new_generation(population, scores, previous_avg_score):
    top_survivors = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)[:int(POPULATION_SIZE * 0.2)]
    new_population = [x[0] for x in top_survivors]
    new_scores = [x[1] for x in top_survivors]
    avg_score = np.mean(new_scores)
    while len(new_population) < POPULATION_SIZE:
        parents = select_parents(population, scores)
        for child in crossover(parents[0], parents[1]):
            child_score = evaluate_gene(child, POPULATION_SIZE)
            new_population.append(mutate(child, child_score, previous_avg_score))
        # child1, child2 = crossover(parents[0], parents[1])
        # child1_score = evaluate_gene(child1, POPULATION_SIZE)
        # child2_score = evaluate_gene(child2, POPULATION_SIZE)
        # new_population.append(mutate(child1, child1_score))
        # new_population.append(mutate(child2, child2_score))
    return new_population[:POPULATION_SIZE], [evaluate_gene(g, POPULATION_SIZE) for g in new_population], avg_score


def round_robin_tournament(genes):
    scores = {str(gene): 0 for gene in genes}
    for i, gene1 in enumerate(genes):
        for gene2 in genes[i + 1:]:
            _, c1, c2 = _PyPacwar.battle(
                list(map(int, gene1)),
                list(map(int, gene2))
            )
            gene1_str, gene2_str = str(gene1), str(gene2)
            if c1 == 0:
                scores[gene2_str] += 5
            elif c2 == 0:
                scores[gene1_str] += 5
            elif c1 > c2:
                scores[gene1_str] += 3
            elif c2 > c1:
                scores[gene2_str] += 3
            elif c1 == c2:
                scores[gene1_str] += 1
                scores[gene2_str] += 1
    # Convert string keys back to lists for sorting and return the sorted gene list based on scores
    sorted_genes = sorted(genes, key=lambda g: scores[str(g)], reverse=True)
    return sorted_genes


def double_elimination_tournament(genes, original_seeds):
    winners = genes[:len(genes) * 2 // 3]
    losers = genes[len(genes) * 2 // 3:]
    winner_round_number = 1
    loser_round_number = 1
    while len(winners) > 1 or len(losers) > 1:
        losers_rounds = []
        if len(winners) > 1:
            # print('winner_round_number:', winner_round_number, ', number of winners:', len(winners))
            winners, new_losers = bracket_round(winners, True, original_seeds)
            losers_rounds += new_losers
            winner_round_number += 1

        if losers_rounds:
            # print('loser_round_number:', loser_round_number, ', number of losers:', len(losers) + len(losers_rounds))
            losers = bracket_round(losers_rounds + losers, False, original_seeds)[0]
            losers = bracket_round(losers, False, original_seeds)[0]
            loser_round_number += 2

    if winners and losers:
        final_winner = compete(winners[0], losers[0])[0]
    else:
        final_winner = winners[0] if winners else losers[0]

    return final_winner


def bracket_round(competitors, is_winners=True, seed_mapping=None):
    sorted_competitors = sorted(competitors, key=lambda x: seed_mapping[str(x)])
    winners = []
    losers = []

    # If odd number, #1 seed gets a bye
    if len(sorted_competitors) % 2 != 0:
        winners.append(sorted_competitors.pop(0))

    # pairs = list(zip(sorted_competitors[:len(sorted_competitors) // 2], reversed(sorted_competitors[len(sorted_competitors) // 2:])))
    # for gene1, gene2 in pairs:
    while len(sorted_competitors) > 1:
        gene1 = sorted_competitors.pop(0)  # Highest seed
        gene2 = sorted_competitors.pop(-1)  # Lowest seed
        winner, loser = compete(gene1, gene2)
        winners.append(winner)
        if is_winners:
            losers.append(loser)

    return winners, losers


def evolution():
    # Main Evolutionary Loop
    pop = [generate_individual() for _ in range(POPULATION_SIZE)]
    evals = [evaluate_gene(gene, POPULATION_SIZE) for gene in pop]
    generation_str_best_genes = []
    generation_best_genes = []
    avg_score = 0
    for g in range(GENERATIONS):
        pop, evals, avg_score = create_new_generation(pop, evals, avg_score)

        # print('%-30s %.50s\n%-30s %.6f' % (
        #     f'Generation #{g + 1} best gene:',
        #     ''.join(map(str, max(zip(pop, evals), key=lambda x: x[1])[0])),
        #     f'With an average score of the top genes in generation #{g + 1}:',
        #     avg_score
        #     )
        # )
        save_best_individual(max(zip(pop, evals), key=lambda x: x[1])[0])
        generation_best_genes.append(max(zip(pop, evals), key=lambda x: x[1])[0])
        generation_str_best_genes.append(''.join(map(str, max(zip(pop, evals), key=lambda x: x[1])[0])))

        # print(f'number of unique genes at generation #{g + 1}: {len(np.unique(generation_str_best_genes))}\n')
    print(f'\nnumber of unique genes from all {GENERATIONS} generations:', len(np.unique(generation_str_best_genes)))

    # Tournament Phase
    seeding_positions = 2 ** int(math.log(2 * GENERATIONS / 3, 2)) + 2 ** int(math.log(GENERATIONS / 3, 2))
    top_genes = round_robin_tournament(generation_best_genes)[:seeding_positions]
    seeds = {str(gene): index for index, gene in enumerate(top_genes)}
    # champion_gene = double_elimination_tournament(top_genes, seeds)

    return double_elimination_tournament(top_genes, seeds)


def run(x):
    if x > 1:
        good_genes = []
        good_genes_str = []
        for i in range(x):
            print('iteration #', i+1, '\n')

            good_gene = evolution()
            good_genes.append(good_gene)
            good_gene_str = ''.join(map(str, good_gene))
            good_genes_str.append(good_gene_str)

            print(f'Best Gene of iteration #{i+1}:', good_gene_str)
            print(f'number of unique genes at iteration {i+1}: {len(np.unique(good_genes_str))}\n')

        print(f'\nnumber of unique genes from all {x} iterations: {len(np.unique(good_genes_str))}\n')
        if x > 2:
            seeding_positions = 2 ** int(math.log(2 * x / 3, 2)) + 2 ** int(math.log(x / 3, 2))
            top_genes = round_robin_tournament(good_genes)[:seeding_positions]
            seeds = {str(gene): index for index, gene in enumerate(top_genes)}
            best_gene = double_elimination_tournament(top_genes, seeds)
            print('Champion Gene:', ''.join(map(str, best_gene)), '\n')
        else:
            _, c1, c2 = _PyPacwar.battle(
                list(map(int, good_genes[0])),
                list(map(int, good_genes[1]))
            )
            best_gene = good_genes[0] if c1 > c2 else good_genes[1]
        return best_gene
    else:
        return evolution()


def test_against_previous_pacwar_gene(candidate_gene, previous):
    if previous == 1:
        pacwar_gene = '33113313011021113313113223323123310123311223020300'
    elif previous == 2:
        pacwar_gene = '01320000133122023333133323222133133222133110301310'
    elif previous == 3:
        pacwar_gene = '01300030103021123333233323223030222032323310200310'
    elif previous == 4:
        pacwar_gene = '00100000111122203003111111111111112111331131111330'
    elif previous == 5:
        pacwar_gene = '01300000110022000133133323332333322233313312232303'
    rounds, c1, c2 = _PyPacwar.battle(list(map(int, candidate_gene)), list(map(int, pacwar_gene)))
    print(f'Test Pacwar {previous} Gene:\nRounds = {rounds}, Candidate Remaining = {c1}, Pacwar Remaining = {c2}\n')


# Constants
X = 12  # Adjust based on desired execution time and experimentation
DIGITS = [0, 1, 2, 3]
POPULATION_SIZE = 50
GENERATIONS = 50
GENE_LENGTH = 50
BASE_RATE = 0.05

pacwar_champ = run(X)
test_against_previous_pacwar_gene(pacwar_champ, 1)
test_against_previous_pacwar_gene(pacwar_champ, 2)
test_against_previous_pacwar_gene(pacwar_champ, 3)
test_against_previous_pacwar_gene(pacwar_champ, 4)
