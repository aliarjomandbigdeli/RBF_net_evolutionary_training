from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as la
import numpy as np
import random
import math

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""


# class ES:
#     def __init__(self, fitness_func):
#         self._population = []
#         self._population_size = 10000
#         self._chromosome_max_size = 10  # in this version length of chromosomes aren't constant
#         self._gene_fields_number = 3  # x,y,r
#         self._tau = 1 / self._population_size ** 0.5
#         self._children = []
#         self._best_chromosome = []
#         self._fitness_func = fitness_func
#
#     def initialize_population(self, max_range, min_range):
#         # chromosome representation : <σ,x1,y1,r1,x2,y2,r2,...>
#         for i in range(self._population_size):
#             chromosome = [random.random() * 300]  # add σ to chromosome
#             for j in range(self._gene_fields_number * random.randint(self._chromosome_max_size)):
#                 chromosome.append(random.random() * (max_range - min_range) + min_range)
#             self._population.append(chromosome)
#
#     def mutation(self):
#         for chromosome in self._population:
#             # mutate σ at first
#             sigma = chromosome[0]
#             sigma = sigma * math.exp(self._tau * random.normalvariate(mu=0, sigma=1))
#             chromosome[0] = sigma
#             # mutate other genes
#             for i in range(1, len(chromosome)):
#                 chromosome[i] = chromosome[i] + sigma * random.normalvariate(mu=0, sigma=1)
#
#     def crossover(self):
#         parent1 = self._population[random.randint(self._population_size)]
#         parent2 = self._population[random.randint(self._population_size)]
#         shorter_parent = parent1
#         longer_parent = parent2
#         if len(longer_parent) < len(shorter_parent):
#             shorter_parent = parent2
#             longer_parent = parent1
#
#         self._children = []
#         for i in range(7 * self._population_size):
#             child = []
#
#             for j in range(len(shorter_parent)):
#                 child.append((shorter_parent[j] + longer_parent[j]) / 2)
#             for j in range(len(shorter_parent), len(longer_parent)):
#                 child.append(longer_parent[j])
#
#             self._children.append(child)
#
#     def fitness(self, chromosome):
#         return self._fitness_func(chromosome)
#
#     def make_wheel(self, population):
#         wheel = []
#         total = sum(self.fitness(p) for p in population)
#         top = 0
#         for p in population:
#             f = self.fitness(p) / total
#             wheel.append((top, top + f, p))
#             top += f
#         return wheel
#
#     def bin_search(self, wheel, num):
#         mid = len(wheel) // 2
#         low, high, answer = wheel[mid]
#         if low <= num <= high:
#             return answer
#         elif low > num:
#             return self.bin_search(wheel[mid + 1:], num)
#         else:
#             return self.bin_search(wheel[:mid], num)
#
#     def select(self, wheel, n_select):
#         """ this method selects chromosome based on SUS(Stochastic Universal Sampling)"""
#         step_size = 1.0 / n_select
#         new_generation = []
#         r = random.random()
#         new_generation.append(self.bin_search(wheel, r))
#         while len(new_generation) < n_select:
#             r += step_size
#             if r > 1:
#                 r %= 1
#             new_generation.append(self.bin_search(wheel, r))
#         return new_generation
#
#     def survivors_selection(self):
#         wheel = self.make_wheel(self._children)
#         self._population = self.select(wheel, self._population_size)
#
#     def exec(self, max_iter):
#         self.initialize_population()
#         for i in range(max_iter):
#             self.mutation()
#             self.crossover()
#             self.survivous_selection()


class RBFClassifier:
    def __init__(self):
        self._data = []
        self._dimension = 2  # number of features
        self._y_star = []
        self._y = []
        self._g = []
        self._w = []  # weight matrix
        # self._gamma = 1.0

        self._min_range = -10
        self._max_range = 10

        self._population = []
        self._population_size = 10000
        self._chromosome_max_size = 10  # in this version length of chromosomes aren't constant
        self._gene_fields_number = 3  # x,y,r
        self._tau = 1 / self._population_size ** 0.5
        self._children = []
        self._best_chromosome = []

    def data(self, d=None):
        """getter and setter of data"""
        if d:
            self._data = d
        return self._data

    def create_random_dataset(self, num_of_data, dimension, cluster_number):
        """create random dataset by normal distribution"""
        x, y = make_blobs(n_samples=num_of_data, centers=cluster_number, n_features=dimension)
        self._dimension = dimension
        self._data = x
        self._y_star = y

        self._max_range = max(self._data[:, 0])
        self._min_range = min(self._data[:, 0])

    def initialize_population(self, max_range, min_range):
        # chromosome representation : <σ,x1,y1,r1,x2,y2,r2,...>
        for i in range(self._population_size):
            chromosome = [random.random() * 300]  # add σ to chromosome
            for j in range(self._gene_fields_number * random.randint(self._chromosome_max_size)):
                chromosome.append(random.random() * (max_range - min_range) + min_range)
            self._population.append(chromosome)

    def mutation(self):
        for chromosome in self._population:
            # mutate σ at first
            sigma = chromosome[0]
            sigma = sigma * math.exp(self._tau * random.normalvariate(mu=0, sigma=1))
            chromosome[0] = sigma
            # mutate other genes
            for i in range(1, len(chromosome)):
                chromosome[i] = chromosome[i] + sigma * random.normalvariate(mu=0, sigma=1)

    def crossover(self):
        parent1 = self._population[random.randint(self._population_size)]
        parent2 = self._population[random.randint(self._population_size)]
        shorter_parent = parent1
        longer_parent = parent2
        if len(longer_parent) < len(shorter_parent):
            shorter_parent = parent2
            longer_parent = parent1

        self._children = []
        for i in range(7 * self._population_size):
            child = []

            for j in range(len(shorter_parent)):
                child.append((shorter_parent[j] + longer_parent[j]) / 2)
            for j in range(len(shorter_parent), len(longer_parent)):
                child.append(longer_parent[j])

            self._children.append(child)

    def fitness(self, chromosome):

        # return self._fitness_func(chromosome)
        return 0

    def make_wheel(self, population):
        wheel = []
        total = sum(self.fitness(p) for p in population)
        top = 0
        for p in population:
            f = self.fitness(p) / total
            wheel.append((top, top + f, p))
            top += f
        return wheel

    def bin_search(self, wheel, num):
        mid = len(wheel) // 2
        low, high, answer = wheel[mid]
        if low <= num <= high:
            return answer
        elif low > num:
            return self.bin_search(wheel[mid + 1:], num)
        else:
            return self.bin_search(wheel[:mid], num)

    def select(self, wheel, n_select):
        """ this method selects chromosome based on SUS(Stochastic Universal Sampling)"""
        step_size = 1.0 / n_select
        new_generation = []
        r = random.random()
        new_generation.append(self.bin_search(wheel, r))
        while len(new_generation) < n_select:
            r += step_size
            if r > 1:
                r %= 1
            new_generation.append(self.bin_search(wheel, r))
        return new_generation

    def survivors_selection(self):
        wheel = self.make_wheel(self._children)
        self._population = self.select(wheel, self._population_size)

    def exec(self, max_iter, data):
        self._data = data
        print("train")
        self.initialize_population(self._max_range, self._min_range)
        for i in range(max_iter):
            self.mutation()
            self.crossover()
            self.survivous_selection()

    def calculate_matrices(self, chromosome):
        g = np.zeros((len(self._data), len(chromosome // self._gene_fields_number)))

        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._gene_fields_number == 1:
                center = [chromosome[i], chromosome[i + 1]]
                # radius_vect = [chromosome[i + 2], chromosome[i + 2]]
                radius_vectors.append(chromosome[i + self._gene_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        self._g = g
        landa = 0.01
        self._w = la.inv(g.transpose().dot(g) + landa * np.identity(len(centers)))
        self._y = g.dot(self._w)

    def train(self, max_iter, data):
        self._data = data
        print("train")
