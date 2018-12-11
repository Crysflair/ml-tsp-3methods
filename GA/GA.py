import random

import matplotlib.pyplot as plt
import numpy as np


class Population:
    def __init__(self, popu_size, gene_len):
        self._popu_size = popu_size
        self._gene_len = gene_len
        self._popu = np.empty((popu_size, gene_len), dtype=np.uint16)
        self._fitness = np.empty(popu_size, dtype=np.float)

    # should be override
    def init_randomly(self):
        pass

    # should be override, update _fitness
    def calculate_fitness(self):
        pass

    # should be override; called by select_crossover
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        pass

    # complete; but need fitness calculated, and _crossover overridden.
    def select_tourna_crossover(self, tourna_size):
        select_range = range(self._popu_size)

        def _tournament(t_size):
            sample_index = random.sample(select_range, t_size)
            sample_fitness = self._fitness[sample_index]
            sample_winner = np.argmax(sample_fitness)
            return self._popu[sample_index[sample_winner]]

        tmp = np.empty_like(self._popu)
        for i in range(self._popu_size):
            p1 = _tournament(tourna_size)
            p2 = _tournament(tourna_size)
            child = self._crossover(p1, p2)
            tmp[i] = child

        self._popu = tmp

    # should be override
    def mutate(self, rate):
        pass

    def get_fittest(self):
        i = np.argmax(self._fitness)
        return self._popu[i], self._fitness[i]


class TSPop(Population):
    def __init__(self, popu_size, city_cnt, dis_mat):
        super().__init__(popu_size, city_cnt)
        self.dis_mat = dis_mat

    def init_randomly(self):
        self._popu[:] = list(range(self._gene_len))
        for i in range(self._popu_size):
            np.random.shuffle(self._popu[i])

    def calculate_fitness(self):
        def calculate_dis(gene):
            dis = 0
            for i in range(self._gene_len-1):
                from_city, to_city = gene[i], gene[i+1]
                dis += self.dis_mat[from_city, to_city]
            from_city, to_city = gene[self._gene_len - 1],  gene[0]
            dis += self.dis_mat[from_city, to_city]
            assert dis > 0
            return dis

        for i in range(self._popu_size):
            self._fitness[i] = 1 / calculate_dis(self._popu[i])

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        # this is TSP crossover method
        start = random.randint(0, self._gene_len - 1)
        end = random.randint(0, self._gene_len-1)
        if end < start:
            start, end = end, start

        child_gene = np.empty(self._gene_len, dtype=np.uint16)
        p1_cross = p1[start:end]
        p2_cross = [x for x in p2 if x not in p1_cross]
        child_gene[0:start], child_gene[start:end], child_gene[end:] = \
            p2_cross[0:start], p1_cross, p2_cross[start:]
        assert child_gene.sum() == p1.sum()
        return child_gene

    def mutate(self, rate):
        for i in range(self._popu_size):
            rand = random.random()
            if rand < rate:
                # mutate
                start = random.randint(0, self._gene_len - 1)
                end = random.randint(0, self._gene_len - 1)
                self._popu[i, start], self._popu[i, end] = self._popu[i, end], self._popu[i, start]
                continue


class GA_TSP_Manager:
    def __init__(self, popu_size, city_cnt, dis_mat, tourna_size, mutate_rate):
        self.tspop = TSPop(popu_size, city_cnt, dis_mat)
        self.tourna_size = tourna_size
        self.mutate_rate = mutate_rate

    def one_iter(self):
        self.tspop.calculate_fitness()
        best_gene, best_value = self.tspop.get_fittest()
        self.tspop.select_tourna_crossover(self.tourna_size)
        self.tspop.mutate(self.mutate_rate)
        return best_gene, best_value

    def run(self, max_iter, max_nochange):
        history_gene = []
        history_value = []
        global_best_value = -1
        no_change = 0
        iter_time = max_iter

        self.tspop.init_randomly()

        for it in range(max_iter):
            best_gene, best_value = self.one_iter()
            history_gene.append(best_gene)
            history_value.append(best_value)

            if best_value > global_best_value:
                global_best_value = best_value
                no_change = 0
            else:
                no_change += 1
            if no_change == max_nochange:
                iter_time = it + 1
                break

        return global_best_value, history_gene, history_value , iter_time


def get_cities(path):
    cities = []
    with open(path) as f:
        cords = f.readlines()
        assert len(cords) == city_cnt
        for cord in cords:
            cord = cord.split()
            cities.append((float(cord[0]), float(cord[1])))
    cities = np.array(cities)
    return cities


def get_distance_matrix(cities: np.ndarray, city_cnt: int) -> np.ndarray:
    distance_mat = np.empty((city_cnt, city_cnt), dtype=np.float32)
    for i in range(city_cnt):
        for j in range(city_cnt):
            distance_mat[i, j] = np.linalg.norm(((cities[i, 0] - cities[j, 0]), cities[i, 1] - cities[j, 1]))
    return distance_mat


if __name__ == '__main__':

    with open('./input_para') as f:
        paras = f.readlines()[1:]

    for para_str in paras:
        para = para_str.split()
        dataset, popu_size, tourna_size, muta_rate, max_iter, max_nochange = \
            para[0], int(para[1]), int(para[2]), float(para[3]), int(para[4]), int(para[5])

        # choose dataset
        if dataset == 'A':
            path = '../test10.txt'
            city_cnt = 10
        elif dataset == 'B':
            path = '../test30.txt'
            city_cnt = 30
        else:
            raise Exception('invalid input format')

        # get cities and distance matrix
        cities = get_cities(path)
        distance_mat = get_distance_matrix(cities, city_cnt)

        # start running
        manager = GA_TSP_Manager(popu_size, city_cnt, distance_mat, tourna_size, muta_rate)
        global_best_value, history_gene, history_value, iter_time = manager.run(max_iter, max_nochange)

        # show and save result
        print('global best value is %f, estimated distance: %f' % (global_best_value, 1/global_best_value))
        print('iter time is:', iter_time)

        fig, ax = plt.subplots()
        ax.plot(range(iter_time), history_value)
        ax.set(xlabel='iter_time', ylabel='fitness', title='testing')
        ax.grid()
        fig.savefig("test.png")
        plt.show()

        best_index = np.argmax(history_value)
        best_gene = history_gene[best_index]
        x = cities[best_gene, 0]
        y = cities[best_gene, 1]

        fig, ax = plt.subplots()
        ax.plot(x, y, 'go-')
        ax.grid()
        ax.axis('equal')
        plt.show()



