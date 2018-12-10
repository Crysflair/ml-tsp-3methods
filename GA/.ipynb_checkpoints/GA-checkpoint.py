import numpy as np

class Population:
    def __init__(self, size, gene_len, is_random_init=True):
        self.size = size
        self.gene_len = gene_len

        self.population = np.zeros((size, gene_len), dtype=np.uint16)
        self.fitness = np.zeros(size, dtype=np.uint16)

        if is_random_init:
            self.population[:] = list(range(gene_len))
            for i in range(size):
                np.random.shuffle(self.population[i])

    def __str__(self):
        print('population:')
        print(self.population)

    def calculate_fitness(self, distance_matrix):
        self.fitness.fill(0)
        for i in range(self.gene_len):


