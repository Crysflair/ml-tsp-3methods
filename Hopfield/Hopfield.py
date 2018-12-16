import matplotlib.pyplot as plt
import numpy as np


# for pre-processing
def get_cities(path):
    cities = []
    with open(path) as f:
        cords = f.readlines()
        for cord in cords:
            cord = cord.split()
            cities.append((float(cord[0]), float(cord[1])))
    cities = np.array(cities)
    return cities


# for pre-processing
def get_distance_matrix(cities: np.ndarray, city_cnt: int) -> np.ndarray:
    distance_mat = np.empty((city_cnt, city_cnt), dtype=np.float32)
    for i in range(city_cnt):
        for j in range(city_cnt):
            distance_mat[i, j] = np.linalg.norm(((cities[i, 0] - cities[j, 0]), cities[i, 1] - cities[j, 1]))
    return distance_mat


# for pre-processing
def read_parameters(path: str)->list:
    with open(path, 'r') as f:
        para_strings = f.readlines()[1:]
    paras = [s.split() for s in para_strings if not s.startswith('#')]
    return paras


class Hopfield_TSP:
    def __init__(self, A, D, city_cnt, distance_mat, g_fun, u0, step):
        self.t = 0
        self.A = A
        self.D = D
        self.city_cnt = city_cnt
        self.distance_mat = distance_mat
        self.u0 = u0
        self.step = step
        if g_fun not in ('sigmoid', 'piecewise'):
            raise AttributeError
        self.g_fun = g_fun

        # 此初始化方法出自“改进的连续Hop网络求解组合优化问题 — — 以 TSP求解为例"
        # 小哥方法的u0前面又乘了一个0.5, 试试！
        delta_u = 2 * np.random.random(city_cnt * city_cnt) - 1   # -1~1
        self.U = 0.5 * u0 * np.log(city_cnt - 1) + delta_u

        # 此连接矩阵和外部输入出自“Hop网络求解TSP的一种改进算法和理论证明”
        self.T = self.set_T()
        self.I = np.ones_like(self.U) * 2 * A
        self.V = np.empty_like(self.U)

    def index2cord(self, index) -> tuple:
        x = index // self.city_cnt
        i = index % self.city_cnt
        return x, i

    def cord_index(self, cord: tuple) -> int:
        x, i = cord
        return x * self.city_cnt + i

    def set_T(self):
        dim = self.city_cnt**2
        T = np.zeros((dim, dim), dtype=np.float)
        for ind1 in range(dim):
            for ind2 in range(dim):
                x, i = self.index2cord(ind1)
                y, j = self.index2cord(ind2)
                if x==y:
                    T[ind1, ind2] -= self.A
                if i==j:
                    T[ind1, ind2] -= self.A
                if j==i+1 or (i==self.city_cnt-1 and j==0):
                    T[ind1, ind2] -= self.D * self.distance_mat[x, y]
        return T

    def activate(self):
        if self.g_fun == 'sigmoid':
            self.V = 0.5 * (1 + np.tanh(self.U / self.u0))
        elif self.g_fun == 'piecewise':
            self.V = self.U + 0.5
            self.V[self.U>0.5] = 1
            self.V[self.U<-0.5] = 0

    def forward(self):
        self.activate()
        dU = self.T.dot(self.V) + self.I
        self.U += dU * self.step
        self.t += 1

    def is_stable(self):
        if ((0.8<self.V)*(self.V<1) + (0<self.V)*(self.V<0.2)).all():
            return True
        else:
            return False

    def calculate_energy(self) -> float:
        E = 0

        part_res = 0
        for x in range(self.city_cnt):
            tmp = 0
            for i in range(self.city_cnt):
                tmp += self.V[self.cord_index((x,i))]
            tmp -= 1
            part_res += tmp**2
        part_res *= 0.5 * self.A
        E += part_res

        part_res = 0
        for i in range(self.city_cnt):
            tmp = 0
            for x in range(self.city_cnt):
                tmp += self.V[self.cord_index((x,i))]
            tmp -= 1
            part_res += tmp**2
        part_res *= 0.5 * self.A
        E += part_res

        part_res = 0
        for x in range(self.city_cnt):
            for y in range(self.city_cnt):
                tmp = 0
                for i in range(city_cnt-1):
                    tmp += self.V[self.cord_index((x,i))] * self.V[self.cord_index((y,i+1))]
                tmp += self.V[self.cord_index((x,city_cnt-1))] * self.V[self.cord_index((y,0))]
                tmp *= self.distance_mat[x,y]
                part_res += tmp
        part_res *= 0.5 * self.D
        E += part_res

        return E

    def run(self, max_iter=500):
        energy_history = []
        while self.t < max_iter:
            self.forward()
            energy_history.append(self.calculate_energy())
            if self.is_stable():
                break
        return self.t, energy_history, self.is_stable()

    def analyze_route(self):
        square_mat = self.V.reshape(self.city_cnt, self.city_cnt)
        route = square_mat.argmax(axis=1)
        test = [x in route for x in range(self.city_cnt)]
        isvalid = True if all(test) else False

        routelen = 0
        for ci in range(self.city_cnt-1):
            from_city, to_city = route[ci], route[ci+1]
            routelen += self.distance_mat[from_city, to_city]
        from_city, to_city = route[self.city_cnt-1], route[0]
        routelen += self.distance_mat[from_city, to_city]

        return route, routelen, isvalid

    def get_neuron_state(self):
        # format: x, i (city, order)
        return self.V.reshape(self.city_cnt, self.city_cnt)


if __name__=='__main__':
    paras = read_parameters('./input_para')
    logfile = './log2.txt'
    output_dir = './results2'

    for para in paras:

        # set parameters
        dataset = para[0]
        A = float(para[1])
        D = float(para[2])
        g_fun = para[3]
        step = float(para[4])
        max_iter = int(para[5])
        repeat = int(para[6])

        if dataset == 'A':
            path = '../test10.txt'
            city_cnt = 10
        elif dataset == 'B':
            path = '../test30.txt'
            city_cnt = 30
        else:
            raise Exception('invalid input format')

        # prepare data
        cities = get_cities(path)
        distance_mat = get_distance_matrix(cities, city_cnt)

        # start working
        with open(logfile, 'a') as f:
            f.write('%s repeat %d times\n' % (str(para), repeat))

        for it in range(repeat):
            hop = Hopfield_TSP(A,D,city_cnt,distance_mat,g_fun,0.02,step)
            iter_cnt, energy_history, is_stable = hop.run(max_iter)
            neuron_state = hop.get_neuron_state()
            route, routelen, isvalid = hop.analyze_route()

            with open(logfile, 'a') as f:
                if isvalid:
                    f.write("valid: %.4f\t%.4f\t%s\t%d\t%s\n" %
                            (routelen, energy_history[-1], str(route), iter_cnt, str(route)))
                else:
                    f.write("invalid: %.4f\t%.4f\t%s\t%d\t%s\n" %
                            (routelen, energy_history[-1], str(route), iter_cnt, str(route)))

            # plot energy curve
            # fig, ax = plt.subplots()
            # ax.plot(range(iter_cnt), energy_history)
            # ax.set(xlabel='iter_time', ylabel='Energy')
            # ax.grid()
            # fig.savefig(output_dir + "/energy_%s(%d).png" % (str(para), it))

            # plot neuron figure
            fig, ax = plt.subplots()
            ax.imshow(neuron_state, cmap='viridis')
            fig.savefig(output_dir + "/grid_%s(%d).png" % (str(para), it))

            # plot route
            fig, ax = plt.subplots()
            x = cities[route, 0]
            y = cities[route, 1]
            ax.plot(x, y, 'go-')
            if isvalid:
                ax.set(title='valid shortest route: %.3f' % (routelen))
            else:
                ax.set(title='invalid shortest route: %.3f' % (routelen))
            ax.grid()
            ax.axis('equal')
            fig.savefig(output_dir + "/route_%s(%d).png" % (str(para), it))

            plt.close('all')
            print('.', end='')















