import numpy as np
import random


class Population:
    def __init__(self, size, gene_len):
        # randomly init population
        self.size = size
        self.gene_len = gene_len
        self.population = np.zeros((size, gene_len), dtype=np.uint16)
        self.fitness = np.zeros(size, dtype=np.uint16)
        self.population[:] = list(range(gene_len))
        for i in range(size):
            np.random.shuffle(self.population[i])

    def __str__(self):
        print('population:')
        print(self.population)

    def calculate_fitness(self, distance_matrix):
        # calculate fitness of the current population. fitness = 1/distance
        def calculate_distance(individual, dist):
            length = 0
            for city_i in range(self.gene_len - 1):
                from_city, to_city = individual[city_i], individual[city_i+1]
                length += dist[from_city, to_city]
            city_i = self.gene_len - 1
            length += dist[individual[city_i], 0]    # from last one to first one
            return length

        for i in range(self.size):
            self.fitness[i] = 1 / calculate_distance(self.population[i], distance_matrix)

    def get_fitness(self):
        return self.fitness

    def get_elite(self, elite_size):
        # return best elite_size individuals, according to fitness
        assert elite_size < self.size // 2
        elite_index = np.argpartition(self.get_fitness(), -elite_size)[-elite_size:]
        return self.population[elite_index]

    def get_best_gene(self, tour: list):
        score = self.fitness[tour]
        max_score_index = np.augmax(score)
        max_popu_index = tour[max_score_index]
        return self.population[max_popu_index]

    def get_population_shape(self):
        return self.size, self.gene_len

    def update_population(self, new_population_array):
        assert (self.size, self.gene_len) == new_population_array
        self.population = new_population_array




class GA:
    def __init__(self, tournament_size, mutation_rate, elitism_size, distance_matrix):
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elite_size = elitism_size
        self.distance_matrix = distance_matrix

    def step(self, population: Population):
        population.calculate_fitness(distance_matrix=self.distance_matrix)

        new_pop_array = self.select_crossover(population)
        self.mutate(new_pop_array)
        if self.elite_size:
            elite_array = population.get_elite(elite_size=self.elite_size)
            new_pop_array[0:self.elite_size] = elite_array

    def select_crossover(self, population: Population):
        def crossover(p1, p2, gene_len):
            start_point = random.randint(0, gene_len-1)
            end_point = random.randint(0, gene_len-1)
            if end_point < start_point:
                start_point, end_point = end_point, start_point

            child_gene = np.empty(gene_len, dtype=np.uint18)
            p1_cross = p1[start_point:end_point]
            p2_cross = [x for x in p2 if x not in p1]

            child_gene[0:start_point], child_gene[start_point:end_point], child_gene[end_point:] = \
                p2_cross[0:start_point], p1_cross, p2_cross[start_point:]

            assert child_gene.sum() == p1.sum()
            return child_gene

        size, gene_len = population.get_population_shape()
        new_pop_array = np.zeros((size, gene_len), dtype=np.uint16)
        select_range = range(size)
        for i in range(size):
            # get parents
            tours_index = random.sample(select_range, self.tournament_size)
            parent1 = population.get_best_gene(tours_index)
            tours_index = random.sample(select_range, self.tournament_size)
            parent2 = population.get_best_gene(tours_index)
            child = crossover(parent1, parent2, gene_len)
            new_pop_array[i] = child
        return new_pop_array

    def mutate(self, new_pop_array: np.ndarray):
        size, gene_len = new_pop_array.shape
        for i in range()









