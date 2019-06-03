from sklearn.datasets.samples_generator import make_blobs
from numpy import linalg as la
import random

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""


class ES:
    def __init__(self):
        self._population = []
        self._population_size = 10000
        self._chromosome_max_size = 10  # in this version length of chromosomes aren't constant
        self._gene_fields_number = 3  # x,y,r
        self._children = []
        self._best_chromosome = []

    def initialize_population(self, max_range, min_range):
        # chromosome representation : <σ,x1,y1,r1,x2,y2,r2,...>
        for i in range(self._population_size):
            chromosome = [random.random() * 300]  # add σ to chromosome
            for j in range(self._gene_fields_number * random.randint(self._chromosome_max_size)):
                chromosome.append(random.random() * (max_range - min_range) + min_range)
            self._population.append(chromosome)

    def mutation(self):
        print("mutation")

    def crossover(self):
        print("parent selection")

    def fitness(self, func, chromosome):
        return func(chromosome)

    def survivors_selection(self):
        for j in range(self._population_size):
            self._population[j] = 0

    def exec(self, max_iter):
        self.initialize_population()
        for i in range(max_iter):
            self.mutation()
            self.crossover()
            self.survivous_selection()


class RBFClassifier:
    def __init__(self):
        self._data = []
        self._dimension = 2  # number of features

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

    def train(self, max_iter, data):
        self._data = data
        print("train")
