import matplotlib.pyplot as plt
import numpy as np


# for pre-processing, return cities
def get_cities(path) -> np.ndarray:
    cities = []
    with open(path) as f:
        cords = f.readlines()
        for cord in cords:
            cord = cord.split()
            cities.append((float(cord[0]), float(cord[1])))
    cities = np.array(cities)
    return cities


# for pre-processing, return distance_mat
def get_distance_matrix(cities: np.ndarray, city_cnt: int) -> np.ndarray:
    distance_mat = np.empty((city_cnt, city_cnt), dtype=np.float32)
    for i in range(city_cnt):
        for j in range(city_cnt):
            distance_mat[i, j] = np.linalg.norm(((cities[i, 0] - cities[j, 0]), cities[i, 1] - cities[j, 1]))
    return distance_mat


# for pre-processing, return paras as list^2
def read_parameters(path: str)->list:
    with open(path, 'r') as f:
        para_strings = f.readlines()[1:]
    paras = [s.split() for s in para_strings if not s.startswith('#')]
    return paras

# for pre-processing
def translatepara(para: list) -> tuple:
    if para[0] == 'A':
        dataset_path = '../test10.txt'
        city_cnt = 10
    elif para[0] == 'B':
        dataset_path = '../test30.txt'
        city_cnt = 30
    else:
        raise AttributeError

    antcnt = int(para[1])
    evap_rate = float(para[2])
    alpha = float(para[3])
    beta = float(para[4])
    max_notchange = int(para[5])
    maxiter = int(para[6])
    repeat = int(para[7])

    return dataset_path, city_cnt, antcnt, evap_rate, alpha, beta, max_notchange, maxiter, repeat

def greedy_pheromone(antcnt, dismat: np.ndarray):
    greedylen = 0
    route = [0]
    cur = 0

    while True:
        good_to_s = np.argsort(dismat[cur])
        for to in tuple(good_to_s):
            if to not in route:
                break
        else:
            break
        greedylen += dismat[cur, to]
        route.append(to)
        cur = to

    return antcnt / greedylen  # test the validity of greedy route!


class Ant:
    def __init__(self, alpha, beta, city_cnt, manager):
        self.alpha = alpha
        self.beta = beta
        self.city_cnt = city_cnt
        self.current = np.random.randint(0, city_cnt)   # [0, city_cnt)
        self.passed = [self.current]
        self.manager = manager

    def reset(self):
        self.current = np.random.randint(0, self.city_cnt)  # [0, city_cnt)
        self.passed = [self.current]

    def _calculate_sub_possiblitily(self, to) -> float:
        if to in self.passed:
            return 0
        else:
            pheromone, heuristic = self.manager.get_path_info(self.current, to)
            return pheromone ** self.alpha * heuristic ** self.beta

    def _calculate_possibility(self) -> np.ndarray:
        pos = np.zeros(self.city_cnt, dtype=np.float)
        for city in range(self.city_cnt):
            sub = self._calculate_sub_possiblitily(city)
            pos[city] = sub
        pos /= pos.sum()
        pos = np.cumsum(pos)
        return  pos

    # select and move to next city
    def select_next_city(self):
        pos = self._calculate_possibility()
        rand = np.random.random()   # [0,1)
        for city in range(self.city_cnt):
            if pos[city] > rand:
                self._move_next_city(city)
                break
        else:
            raise Exception("unable to select next city!")

    def _move_next_city(self, city):
        self.passed.append(city)
        self.current = city

    def get_route(self):
        return self.passed


class MapManager:
    def __init__(self, distant_mat, init_pheromone, evap_rate, Q):
        self.distant_map = distant_mat
        self.heuristic_map = 1/distant_mat
        self.init_pheromone = init_pheromone
        self.pheromone_map = np.ones_like(self.heuristic_map) * self.init_pheromone
        self.evap_rate = evap_rate
        self.Q = Q

    def reset(self):
        self.pheromone_map = np.ones_like(self.heuristic_map) * self.init_pheromone

    # this is called by ants
    def get_path_info(self,fro, to) -> tuple:
        return self.pheromone_map[fro, to], self.heuristic_map[fro, to]

    def evaporate(self):
        self.pheromone_map *= 1 - self.evap_rate

    def increase_ant(self, ant: Ant):
        route = ant.get_route()

        assert len(route) == ant.city_cnt
        test = [x in route for x in range(len(route))]
        isvalid = True if all(test) else False
        assert isvalid

        routelen = 0
        route.append(route[0])

        # calculate the overall length of this ant's trip
        for i in range(len(route)-1):
            fro, to = route[i], route[i+1]
            routelen += self.distant_map[fro, to]

        # add pheromone to where it passed
        increment = self.Q / routelen
        for i in range(len(route)-1):
            fro, to = route[i], route[i+1]
            self.pheromone_map[fro, to] += increment

        # return the length of route for further use
        return route[:-1], routelen


class ACO:
    def __init__(self, max_notchange, manager: MapManager, ants, city_cnt, maxiter):
        self.max_notchange = max_notchange
        self.manager = manager
        self.ants = ants
        self.city_cnt = city_cnt
        self.maxiter = maxiter

    # return bestroute, bestlen
    def _one_iter(self):
        for ant in self.ants:
            ant.reset()

        # start the trip
        for step in range(self.city_cnt-1):
            for ant in self.ants:
                ant.select_next_city()

        # update pheromone
        self.manager.evaporate()

        # find best solution among all ants
        # & update pheromone for each ant (increment)
        bestroute = None
        bestlen = float('inf')
        for ant in self.ants:
            route, routelen = self.manager.increase_ant(ant)
            if routelen < bestlen:
                bestlen = routelen
                bestroute = route

        return bestroute, bestlen

    def run(self):
        gbestroute = None
        gbestlen = float('inf')
        history = []
        nochange = 0
        self.manager.reset()

        for it in range(self.maxiter):
            bestroute, bestlen = self._one_iter()
            history.append(bestlen)
            if bestlen < gbestlen:
                nochange = 0
                gbestlen = bestlen
                gbestroute = bestroute
            else:
                nochange += 1
            if nochange > self.max_notchange:
                break
        return history, gbestroute, gbestlen


if __name__ == '__main__':
    paras = read_parameters('./input_para')
    logfile = './log.txt'
    output_dir = './results2'

    for para in paras:

        # variables in this experiment: alpha, beta, evap, Q, antcnt.
        dataset_path, city_cnt, antcnt, evap_rate, alpha, beta, max_notchange, maxiter, repeat = translatepara(para)
        assert evap_rate < 1
        cities = get_cities(dataset_path)
        dismat = get_distance_matrix(cities, city_cnt)

        # init components
        init_pheromone = greedy_pheromone(antcnt, dismat)
        Q = 1
        dismat[dismat == 0] = float('inf')
        manager = MapManager(distant_mat=dismat, init_pheromone=init_pheromone, evap_rate=evap_rate, Q=Q)
        ants = []
        for _ in range(antcnt):
            ants.append(Ant(alpha=alpha, beta=beta, city_cnt=city_cnt, manager=manager))
        aco = ACO(max_notchange=max_notchange, manager=manager, ants=ants, city_cnt=city_cnt, maxiter=maxiter)

        # start experiment
        with open(logfile, 'a') as f:
            f.write('%s repeat %d times\n' % (str(para), repeat))

        for it in range(repeat):
            history, route, routelen = aco.run()
            itertime = len(history)
            with open(logfile, 'a') as f:
                f.write("%.4f\t%s\t%d\n" % (routelen, str(route), itertime))

            # plot route
            fig, ax = plt.subplots()
            x = cities[route, 0]
            y = cities[route, 1]
            ax.plot(x, y, 'go-')
            ax.set(title='Shortest route: %.4f' % routelen)
            ax.grid()
            ax.axis('equal')
            fig.savefig(output_dir + "/route_%s(%d).png" % (str(para), it))

            # plot curve
            fig, ax = plt.subplots()
            ax.plot(range(itertime), history)
            ax.set(xlabel='iter_time', ylabel='Shortest route')
            ax.grid()
            fig.savefig(output_dir + "/length_%s(%d).png" % (str(para), it))

            plt.close('all')
            print('.', end='')







