from sklearn.datasets.samples_generator import make_blobs, make_regression
from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""


class RBFRegression:
    def __init__(self):
        self._data = []
        self._data_test = []
        self._dimension = 2  # number of features
        self._y_star = []
        self._y_star_test = []
        self._y = []
        self._y_test = []
        self._g = []
        self._w = []  # weight matrix

        # self._min_range = -10
        # self._max_range = 10

        self._population = []
        self._mutated_population = []
        self._population_size = 30
        self._child2population_ratio = 4
        self._chromosome_max_bases = 8  # in this version length of chromosomes aren't constant
        self._chromosome_min_bases = 4
        self._base_fields_number = 4  # x,r (dimension + 1(for radius))
        self._tau = 0.5 / ((self._base_fields_number * self._chromosome_max_bases) ** 0.5)
        self._children = []
        self._best_chromosome = []
        self._best_fitness_list = [0]
        self._avg_fitness_list = [0]
        self._range_mat = []
        self._most_dist = 0.0

    def data(self, d=None):
        """getter and setter of data"""
        if d:
            self._data = d
        return self._data

    def y(self):
        """:returns predicted vector"""
        return self._y

    def y_star(self):
        return self._y_star

    def create_random_dataset(self, num_of_data, dimension):
        x = np.random.uniform(0., 2., num_of_data)
        x = np.sort(x, axis=0)
        noise = np.random.uniform(-0.05, 0.05, num_of_data)
        y = np.sin(2 * np.pi * x) + noise

        # x, y = make_regression(n_samples=num_of_data, n_features=dimension, noise=0.1)
        self._dimension = dimension
        self._data = x
        self._y_star = y

    def read_excel(self, train_address):
        dataset_train = pd.read_excel(train_address)
        self._data = dataset_train.iloc[:, 0:dataset_train.shape[1] - 1].values
        print(self._data)
        # dataset_length = self._data.shape[0]
        self._dimension = self._data.shape[1]
        self._y_star = dataset_train.iloc[:, dataset_train.shape[1] - 1:dataset_train.shape[1]].values
        self._y_star = self._y_star[:, 0]
        print(f'y star len: {len(self._y_star)}')
        print(f'y star shape: {self._y_star.shape}')
        print(self._y_star)

        self._data_test = self._data
        self._y_star_test = self._y_star

        random_indexes = np.random.randint(0, len(self._data_test), int(0.6 * len(self._data_test)))

        tmp_list = []
        tmp_list2 = []
        for i in random_indexes:
            tmp_list.append(self._data_test[i])
            tmp_list2.append(self._y_star_test[i])
        self._data = np.array(tmp_list)
        self._y_star = np.array(tmp_list2)

    def initialize_parameters_based_on_data(self):
        self._base_fields_number = self._dimension + 1

        self._tau = 0.5 / ((self._base_fields_number * self._chromosome_max_bases) ** 0.5)
        # self._tau = 1 / (self._base_fields_number ** 0.5)

        self._range_mat = np.zeros((self._dimension, 2))
        if self._dimension > 1:  # if just for random data(it should remove when file reading)
            for i in range(self._dimension):
                self._range_mat[i, 0] = np.max(self._data[:, i])
                self._range_mat[i, 1] = np.min(self._data[:, i])
        else:
            self._range_mat[0, 0] = np.max(self._data)
            self._range_mat[0, 1] = np.min(self._data)
        # s = 0.0
        # for i in range(self._dimension):
        #     s += (self._range_mat[i, 0] - self._range_mat[i, 1]) ** 2
        # self._most_dist = s ** (1 / self._dimension)

        self._most_dist = np.max(self._data) - np.min(self._data)

        # print(f'range mat: {self._range_mat}')
        # print(f'most distance: {self._most_dist}')

    def initialize_population(self):
        # chromosome representation : <σ,x1,y1,r1,x2,y2,r2,...>
        for i in range(self._population_size):
            chromosome = [np.random.uniform(self._most_dist * 0.01, self._most_dist * 0.1)]  # add σ to chromosome
            for j in range(
                    self._base_fields_number * random.randint(self._chromosome_min_bases, self._chromosome_max_bases)):
                if (j + 1) % self._base_fields_number != 0:
                    chromosome.append(random.random() * (
                            self._range_mat[j % self._base_fields_number, 0] - self._range_mat[
                        j % self._base_fields_number, 1]) + self._range_mat[j % self._base_fields_number, 1])
                    # chromosome.append(random.random() * (max_range - min_range) + min_range)
                else:  # radius can't be negative
                    chromosome.append(random.random() * self._most_dist)
            # print(f'chromosome {i}: {chromosome}, len: {len(chromosome)}')
            self._population.append(np.array(chromosome))

    def mutation(self):
        self._mutated_population = []
        for chromosome in self._population:
            mutated_chromosome = np.copy(chromosome)
            # mutate σ at first
            sigma = mutated_chromosome[0] * math.exp(self._tau * np.random.normal(0, 1))
            # print(f'past sigma: {chromosome[0]}, new sigma: {sigma}')
            mutated_chromosome[0] = sigma
            # mutate other genes
            for i in range(1, len(chromosome)):
                mutated_chromosome[i] += sigma * np.random.normal(0, 1)
            self._mutated_population.append(mutated_chromosome)
            # print(f'mutated chromosome: {mutated_chromosome}')

    def crossover(self):
        for i in range(self._child2population_ratio * self._population_size):
            parent1 = self._mutated_population[random.randint(0, self._population_size - 1)]
            parent2 = self._mutated_population[random.randint(0, self._population_size - 1)]
            if random.uniform(.0, 1.) < 0.4:
                shorter_parent = parent1
                longer_parent = parent2
                if len(longer_parent) < len(shorter_parent):
                    shorter_parent = parent2
                    longer_parent = parent1

                child = (shorter_parent + longer_parent[:len(shorter_parent)]) / 2
                if random.uniform(.0, 1.) >= 0.5:
                    child = np.append(child, longer_parent[len(shorter_parent):])
            else:
                if random.uniform(.0, 1.) > 0.5:
                    child = parent1
                else:
                    child = parent2
            # print(f'child {child}, fitness {self.fitness(child)}')
            self._children.append(child)

    def select_best(self, chromosome_list):
        bst = chromosome_list[0]
        bst_fit = self.fitness(bst)
        for i in chromosome_list:
            fit_i = self.fitness(i)
            # print(f'fitness: {fit_i}')
            if bst_fit < fit_i:
                bst = i
                bst_fit = fit_i
            elif bst_fit == fit_i and len(i) < len(bst):
                bst = i
                bst_fit = fit_i
        return bst

    def return_best_avg_fit(self, chromosome_list):
        s = 0
        bst = chromosome_list[0]
        bst_fit = self.fitness(bst)
        for i in chromosome_list:
            fit_i = self.fitness(i)
            s += fit_i
            if bst_fit < fit_i:
                bst = i
                bst_fit = fit_i
        return self.fitness(bst), s / len(chromosome_list)

    def survivors_selection(self):
        """ this method works based on q-tournament """
        q = 5
        new_population = []
        for i in range(self._population_size):
            batch = []
            for j in range(q):
                r = random.randint(0, (self._child2population_ratio + 1) * self._population_size - 1)
                if r < self._population_size:
                    batch.append(self._population[r])
                else:
                    batch.append(self._children[r - self._population_size])
            new_population.append(self.select_best(batch))

        self._population = new_population

    def train(self, max_iter, data):
        self._data = data

        self.initialize_population()
        for i in range(max_iter):
            self.mutation()
            self.crossover()
            self.survivors_selection()
            print(f'iter {i}')
            bst, avg = self.return_best_avg_fit(self._population)
            self._best_fitness_list.append(bst)
            self._avg_fitness_list.append(avg)

        self._best_chromosome = self.select_best(self._population)
        print(f'best : {self._best_chromosome}')
        print(self.fitness(self._best_chromosome))  # just for updating y

    def fitness(self, chromosome):
        self.calculate_matrices(chromosome)
        error = 0.5 * (self._y - self._y_star).transpose().dot(self._y - self._y_star)
        # error = 0.5 * (la.norm(self._y - self._y_star, 2) ** 2)
        return 1 / error

    def calculate_matrices(self, chromosome):
        g = np.zeros((len(self._data), len(chromosome) // self._base_fields_number))

        # print(f'fitness chromosome: {chromosome}, len: {len(chromosome)}')
        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._base_fields_number == 1:
                center = chromosome[i: i + self._dimension]
                radius_vectors.append(chromosome[i + self._base_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        self._g = g
        lam = 0.001
        self._w = la.inv(g.transpose().dot(g) + lam * np.identity(len(centers))).dot(g.transpose()).dot(self._y_star)
        self._y = g.dot(self._w)

        # print(f'y_star type{type(self._y_star)}')
        # print(f'y type{type(self._y)}')

    def test(self):
        chromosome = self._best_chromosome
        g = np.zeros((len(self._data_test), len(chromosome) // self._base_fields_number))

        # print(f'fitness chromosome: {chromosome}, len: {len(chromosome)}')
        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._base_fields_number == 1:
                center = chromosome[i: i + self._dimension]
                radius_vectors.append(chromosome[i + self._base_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data_test)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data_test[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        self._y_test = g.dot(self._w)

        error = 0.5 * (self._y - self._y_star).transpose().dot(self._y - self._y_star)
        print(f'error test: {error}')


class RBFBinClassifier:
    def __init__(self):
        self._data = []
        self._data_test = []
        self._dimension = 2  # number of features
        self._y_star = []
        self._y_star_test = []
        self._y = []
        self._y_test = []
        self._g = []
        self._w = []  # weight matrix

        self._population = []
        self._mutated_population = []
        self._population_size = 30
        self._child2population_ratio = 7
        self._chromosome_max_bases = 14  # in this version length of chromosomes aren't constant
        self._chromosome_min_bases = 7
        self._base_fields_number = 2  # x,r (dimension + 1(for radius))
        self._tau = 1 / (self._base_fields_number ** 0.5)
        self._children = []
        self._best_chromosome = []
        self._best_fitness_list = [0]
        self._avg_fitness_list = [0]
        self._range_mat = []
        self._most_dist = 0.0

    def data(self, d=None):
        """getter and setter of data"""
        if d:
            self._data = d
        return self._data

    def y(self):
        """:returns predicted vector"""
        return self._y

    def y_star(self):
        return self._y_star

    def create_random_dataset(self, num_of_data, cluster_number, dimension):
        """create random dataset by normal distribution"""
        x, y = make_blobs(n_samples=num_of_data, centers=cluster_number, n_features=dimension)
        self._dimension = dimension
        self._data = x
        self._y_star = y

    def read_excel(self, train_address, test_address=None):
        dataset_train = pd.read_excel(train_address)
        self._data = dataset_train.iloc[:, 0:dataset_train.shape[1] - 1].values
        print(self._data)
        # dataset_length = self._data.shape[0]
        self._dimension = self._data.shape[1]
        self._y_star = dataset_train.iloc[:, dataset_train.shape[1] - 1:dataset_train.shape[1]].values
        self._y_star = self._y_star[:, 0]
        print(f'y star len: {len(self._y_star)}')
        print(f'y star shape: {self._y_star.shape}')
        print(self._y_star)

        if test_address is not None:
            dataset_test = pd.read_excel(train_address)
            self._data_test = dataset_test.iloc[:, 0:dataset_test.shape[1] - 1].values
            self._y_star_test = dataset_test.iloc[:, dataset_test.shape[1] - 1:dataset_test.shape[1]].values
            self._y_star_test = self._y_star_test[:, 0]

            if np.min(self._y_star_test) == -1:
                self._y_star_test = 0.5 * self._y_star_test + 0.5

    def initialize_parameters_based_on_data(self):
        self._base_fields_number = self._dimension + 1

        # self._tau = 0.5 / ((self._base_fields_number * self._chromosome_max_bases) ** 0.5)
        self._tau = 1 / (self._base_fields_number ** 0.5)

        self._range_mat = np.zeros((self._dimension, 2))
        for i in range(self._dimension):
            self._range_mat[i, 0] = np.max(self._data[:, i])
            self._range_mat[i, 1] = np.min(self._data[:, i])
        # s = 0.0
        # for i in range(self._dimension):
        #     s += (self._range_mat[i, 0] - self._range_mat[i, 1]) ** 2
        # self._most_dist = s ** (1 / self._dimension)

        self._most_dist = np.max(self._data) - np.min(self._data)

        if np.min(self._y_star) == -1:
            self._y_star = 0.5 * self._y_star + 0.5

        print(f'y_star new: {self._y_star}')
        # print(f'range mat: {self._range_mat}')
        # print(f'most distance: {self._most_dist}')

    def initialize_population(self):
        m = len(self._data * self._dimension) ** (1 / self._dimension)
        # chromosome representation : <σ,x1,y1,r1,x2,y2,r2,...>
        for i in range(self._population_size):
            chromosome = [np.random.uniform(self._most_dist * 0.01, self._most_dist * 0.1)]  # add σ to chromosome
            for j in range(
                    self._base_fields_number * random.randint(self._chromosome_min_bases, self._chromosome_max_bases)):
                if (j + 1) % self._base_fields_number != 0:
                    chromosome.append(random.random() * (
                            self._range_mat[j % self._base_fields_number, 0] - self._range_mat[
                        j % self._base_fields_number, 1]) + self._range_mat[j % self._base_fields_number, 1])
                    # chromosome.append(random.random() * (max_range - min_range) + min_range)
                else:  # radius can't be negative
                    chromosome.append(random.uniform(0.75 * m, m))
                    # chromosome.append(random.random() * self._most_dist)
            # print(f'chromosome {i}: {chromosome}, len: {len(chromosome)}')
            self._population.append(np.array(chromosome))

    def mutation(self):
        self._mutated_population = []
        for chromosome in self._population:
            mutated_chromosome = np.copy(chromosome)
            # mutate σ at first
            sigma = mutated_chromosome[0] * math.exp(self._tau * np.random.normal(0, 1))
            # print(f'past sigma: {chromosome[0]}, new sigma: {sigma}')
            mutated_chromosome[0] = sigma
            # mutate other genes
            for i in range(1, len(chromosome)):
                mutated_chromosome[i] += sigma * np.random.normal(0, 1)
            self._mutated_population.append(mutated_chromosome)
            # print(f'mutated chromosome: {mutated_chromosome}')

    def crossover(self):
        for i in range(self._child2population_ratio * self._population_size):
            parent1 = self._mutated_population[random.randint(0, self._population_size - 1)]
            parent2 = self._mutated_population[random.randint(0, self._population_size - 1)]
            if random.uniform(.0, 1.) < 0.4:
                shorter_parent = parent1
                longer_parent = parent2
                if len(longer_parent) < len(shorter_parent):
                    shorter_parent = parent2
                    longer_parent = parent1

                child = (shorter_parent + longer_parent[:len(shorter_parent)]) / 2
                if random.uniform(.0, 1.) >= 0.5:
                    child = np.append(child, longer_parent[len(shorter_parent):])
            else:
                if random.uniform(.0, 1.) > 0.5:
                    child = parent1
                else:
                    child = parent2
            # print(f'child {child}, fitness {self.fitness(child)}')
            self._children.append(child)

    def select_best(self, chromosome_list):
        bst = chromosome_list[0]
        bst_fit = self.fitness(bst)
        for i in chromosome_list:
            fit_i = self.fitness(i)
            # print(f'fitness: {fit_i}')
            if bst_fit < fit_i:
                bst = i
                bst_fit = fit_i
            elif bst_fit == fit_i and len(i) < len(bst):
                bst = i
                bst_fit = fit_i
        return bst

    def return_best_avg_fit(self, chromosome_list):
        s = 0
        bst = chromosome_list[0]
        bst_fit = self.fitness(bst)
        for i in chromosome_list:
            fit_i = self.fitness(i)
            s += fit_i
            if bst_fit < fit_i:
                bst = i
                bst_fit = fit_i
        return self.fitness(bst), s / len(chromosome_list)

    def survivors_selection(self):
        """ this method works based on q-tournament """
        q = 6
        new_population = []
        for i in range(self._population_size):
            batch = []
            for j in range(q):
                r = random.randint(0, (self._child2population_ratio + 1) * self._population_size - 1)
                if r < self._population_size:
                    batch.append(self._population[r])
                else:
                    batch.append(self._children[r - self._population_size])
            new_population.append(self.select_best(batch))

        self._population = new_population

    def train(self, max_iter, data):
        self._data = data

        self.initialize_population()
        for i in range(max_iter):
            self.mutation()
            self.crossover()
            self.survivors_selection()
            print(f'iter {i}')
            # bst, avg = self.return_best_avg_fit(self._population)
            # self._best_fitness_list.append(bst)
            # self._avg_fitness_list.append(avg)

        self._best_chromosome = self.select_best(self._population)
        print(f'best : {self._best_chromosome}')
        print(self.fitness(self._best_chromosome))  # just for updating y

    def fitness(self, chromosome):
        self.calculate_matrices(chromosome)
        return 1 - np.sum(np.abs(0.5 * np.sign(self._y) + 0.5 - self._y_star)) / len(self._data)
        # return 1 - np.sum(np.abs(np.sign(self._y) - self._y)) / len(self._data)

    def calculate_matrices(self, chromosome):
        g = np.zeros((len(self._data), len(chromosome) // self._base_fields_number))

        # print(f'fitness chromosome: {chromosome}, len: {len(chromosome)}')
        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._base_fields_number == 1:
                center = chromosome[i: i + self._dimension]
                radius_vectors.append(chromosome[i + self._base_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        self._g = g
        lam = 0.001
        self._w = la.inv(g.transpose().dot(g) + lam * np.identity(len(centers))).dot(g.transpose()).dot(self._y_star)
        self._y = g.dot(self._w)

    def test(self):
        chromosome = self._best_chromosome
        g = np.zeros((len(self._data_test), len(chromosome) // self._base_fields_number))

        # print(f'fitness chromosome: {chromosome}, len: {len(chromosome)}')
        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._base_fields_number == 1:
                center = chromosome[i: i + self._dimension]
                radius_vectors.append(chromosome[i + self._base_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data_test)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data_test[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        # lam = 0.001
        # w = la.inv(g.transpose().dot(g) + lam * np.identity(len(centers))).dot(g.transpose()).dot(self._y_star_test)
        self._y_test = g.dot(self._w)
        print(f'data test shape {self._data_test.shape}')
        print(f'y star test shape {self._y_star_test.shape}')
        print(f'y test shape {len(self._y_test.shape)}')

        accuracy = 1 - np.sum(np.abs(0.5 * np.sign(np.around(self._y_test)) + 0.5 - self._y_star_test)) / len(
            self._data_test)
        print(f'accuracy test: {accuracy}')


class RBFClassifier:
    def __init__(self):
        self._data = []
        self._data_test = []
        self._dimension = 2  # number of features
        self._y_star = []
        self._y_star_test = []
        self._y_star_before_1hot = []
        self._y_star_test_before_1hot = []
        self._y = []
        self._y_test = []
        self._g = []
        self._w = []  # weight matrix

        self._population = []
        self._mutated_population = []
        self._population_size = 30
        self._child2population_ratio = 7
        self._chromosome_max_bases = 6  # in this version length of chromosomes aren't constant
        self._chromosome_min_bases = 5
        self._base_fields_number = 2  # x,r (dimension + 1(for radius))
        self._tau = 1 / (self._base_fields_number ** 0.5)
        self._children = []
        self._best_chromosome = []
        self._best_fitness_list = [0]
        self._avg_fitness_list = [0]
        self._range_mat = []
        self._most_dist = 0.0
        self._num_classes = 5

    def data(self, d=None):
        """getter and setter of data"""
        if d:
            self._data = d
        return self._data

    def y(self):
        """:returns predicted vector"""
        return self._y

    def y_star(self):
        return self._y_star

    def create_random_dataset(self, num_of_data, cluster_number, dimension):
        """create random dataset by normal distribution"""
        x, y = make_blobs(n_samples=num_of_data, centers=cluster_number, n_features=dimension)
        self._dimension = dimension
        self._num_classes = cluster_number
        self._data = x
        self._y_star = y

    def read_excel(self, train_address, test_address=None):
        dataset_train = pd.read_excel(train_address)
        self._data = dataset_train.iloc[:, 0:dataset_train.shape[1] - 1].values
        print(self._data)
        # dataset_length = self._data.shape[0]
        self._dimension = self._data.shape[1]
        self._y_star = dataset_train.iloc[:, dataset_train.shape[1] - 1:dataset_train.shape[1]].values
        self._y_star = self._y_star[:, 0]
        print(f'y star len: {len(self._y_star)}')
        print(f'y star shape: {self._y_star.shape}')
        print(self._y_star)

        if test_address is not None:
            dataset_test = pd.read_excel(train_address)
            self._data_test = dataset_test.iloc[:, 0:dataset_test.shape[1] - 1].values
            self._y_star_test = dataset_test.iloc[:, dataset_test.shape[1] - 1:dataset_test.shape[1]].values
            self._y_star_test = self._y_star_test[:, 0]
            self.one_hot_y_star_test()

    def one_hot_y_star_test(self):
        if np.min(self._y_star_test) == 1:
            self._y_star_test -= 1
        if np.min(self._y_star_test) == -1:
            self._y_star_test = 0.5 * self._y_star_test + 0.5

        self._y_star_test_before_1hot = self._y_star_test
        # self._y_star_test_before_1hot = np.around(self._y_star_test_before_1hot)

        self._y_star_test = np.zeros((len(self._y_star_test_before_1hot), self._num_classes))
        self._y_star_test[np.arange(len(self._y_star_test_before_1hot)), self._y_star_test_before_1hot] = 1

    def one_hot(self):
        if np.min(self._y_star) == 1:
            self._y_star -= 1
        if np.min(self._y_star) == 1:
            self._y_star = 0.5 * self._y_star + 0.5

        print(f'y star in one hot : {self._y_star}')
        self._y_star_before_1hot = self._y_star
        self._y_star = np.zeros((len(self._y_star_before_1hot), self._num_classes))
        self._y_star[np.arange(len(self._y_star_before_1hot)), self._y_star_before_1hot] = 1

    def initialize_parameters_based_on_data(self):
        self._base_fields_number = self._dimension + 1

        # self._tau = 0.5 / ((self._base_fields_number * self._chromosome_max_bases) ** 0.5)
        self._tau = 1 / (self._base_fields_number ** 0.5)

        self._range_mat = np.zeros((self._dimension, 2))
        for i in range(self._dimension):
            self._range_mat[i, 0] = np.max(self._data[:, i])
            self._range_mat[i, 1] = np.min(self._data[:, i])
        # s = 0.0
        # for i in range(self._dimension):
        #     s += (self._range_mat[i, 0] - self._range_mat[i, 1]) ** 2
        # self._most_dist = s ** (1 / self._dimension)

        self._most_dist = np.max(self._data) - np.min(self._data)

        self.one_hot()
        # print(f'range mat: {self._range_mat}')
        # print(f'most distance: {self._most_dist}')

    def initialize_population(self):
        # chromosome representation : <σ,x1,y1,r1,x2,y2,r2,...>
        for i in range(self._population_size):
            chromosome = [np.random.uniform(self._most_dist * 0.01, self._most_dist * 0.1)]  # add σ to chromosome
            for j in range(
                    self._base_fields_number * random.randint(self._chromosome_min_bases, self._chromosome_max_bases)):
                if (j + 1) % self._base_fields_number != 0:
                    chromosome.append(random.random() * (
                            self._range_mat[j % self._base_fields_number, 0] - self._range_mat[
                        j % self._base_fields_number, 1]) + self._range_mat[j % self._base_fields_number, 1])
                    # chromosome.append(random.random() * (max_range - min_range) + min_range)
                else:  # radius can't be negative
                    chromosome.append(random.random() * self._most_dist)
            # print(f'chromosome {i}: {chromosome}, len: {len(chromosome)}')
            self._population.append(np.array(chromosome))

    def mutation(self):
        self._mutated_population = []
        for chromosome in self._population:
            mutated_chromosome = np.copy(chromosome)
            # mutate σ at first
            sigma = mutated_chromosome[0] * math.exp(self._tau * np.random.normal(0, 1))
            # print(f'past sigma: {chromosome[0]}, new sigma: {sigma}')
            mutated_chromosome[0] = sigma
            # mutate other genes
            for i in range(1, len(chromosome)):
                mutated_chromosome[i] += sigma * np.random.normal(0, 1)
            self._mutated_population.append(mutated_chromosome)
            # print(f'mutated chromosome: {mutated_chromosome}')

    def crossover(self):
        for i in range(self._child2population_ratio * self._population_size):
            parent1 = self._mutated_population[random.randint(0, self._population_size - 1)]
            parent2 = self._mutated_population[random.randint(0, self._population_size - 1)]
            if random.uniform(.0, 1.) < 0.4:
                shorter_parent = parent1
                longer_parent = parent2
                if len(longer_parent) < len(shorter_parent):
                    shorter_parent = parent2
                    longer_parent = parent1

                child = (shorter_parent + longer_parent[:len(shorter_parent)]) / 2
                if random.uniform(.0, 1.) >= 0.5:
                    child = np.append(child, longer_parent[len(shorter_parent):])
            else:
                if random.uniform(.0, 1.) > 0.5:
                    child = parent1
                else:
                    child = parent2
            # print(f'child {child}, fitness {self.fitness(child)}')
            self._children.append(child)

    def select_best(self, chromosome_list):
        bst = chromosome_list[0]
        bst_fit = self.fitness(bst)
        for i in chromosome_list:
            fit_i = self.fitness(i)
            # print(f'fitness: {fit_i}')
            if bst_fit < fit_i:
                bst = i
                bst_fit = fit_i
            elif bst_fit == fit_i and len(i) < len(bst):
                bst = i
                bst_fit = fit_i
        return bst

    def return_best_avg_fit(self, chromosome_list):
        s = 0
        bst = chromosome_list[0]
        bst_fit = self.fitness(bst)
        for i in chromosome_list:
            fit_i = self.fitness(i)
            s += fit_i
            if bst_fit < fit_i:
                bst = i
                bst_fit = fit_i
        return self.fitness(bst), s / len(chromosome_list)

    def survivors_selection(self):
        """ this method works based on q-tournament """
        q = 6
        new_population = []
        for i in range(self._population_size):
            batch = []
            for j in range(q):
                r = random.randint(0, (self._child2population_ratio + 1) * self._population_size - 1)
                if r < self._population_size:
                    batch.append(self._population[r])
                else:
                    batch.append(self._children[r - self._population_size])
            new_population.append(self.select_best(batch))

        self._population = new_population

    def train(self, max_iter, data):
        self._data = data

        self.initialize_population()
        for i in range(max_iter):
            self.mutation()
            self.crossover()
            self.survivors_selection()
            print(f'iter {i}')
            # bst, avg = self.return_best_avg_fit(self._population)
            # self._best_fitness_list.append(bst)
            # self._avg_fitness_list.append(avg)

        self._best_chromosome = self.select_best(self._population)
        print(f'best : {self._best_chromosome}')
        print(self.fitness(self._best_chromosome))  # just for updating y
        self._y = np.argmax(self._y, axis=1)

    def fitness(self, chromosome):
        self.calculate_matrices(chromosome)
        return 1 - np.sum(np.sign(np.abs(np.argmax(self._y, axis=1) - np.argmax(self._y_star, axis=1)))) / len(
            self._data)

    def calculate_matrices(self, chromosome):
        g = np.zeros((len(self._data), len(chromosome) // self._base_fields_number))

        # print(f'fitness chromosome: {chromosome}, len: {len(chromosome)}')
        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._base_fields_number == 1:
                center = chromosome[i: i + self._dimension]
                radius_vectors.append(chromosome[i + self._base_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        self._g = g
        lam = 0.001
        self._w = la.inv(g.transpose().dot(g) + lam * np.identity(len(centers))).dot(g.transpose()).dot(self._y_star)
        self._y = g.dot(self._w)

    def test(self):
        chromosome = self._best_chromosome
        g = np.zeros((len(self._data_test), len(chromosome) // self._base_fields_number))
        # print(f'fitness chromosome: {chromosome}, len: {len(chromosome)}')
        centers = []
        radius_vectors = []
        for i in range(len(chromosome)):
            if i % self._base_fields_number == 1:
                center = chromosome[i: i + self._dimension]
                radius_vectors.append(chromosome[i + self._base_fields_number - 1])
                centers.append(center)

        for i in range(len(self._data_test)):
            for j in range(len(centers)):
                g[i, j] = math.exp(-1 * (la.norm(self._data_test[i] - centers[j], 2) / radius_vectors[j]) ** 2)

        # self._g = g
        # lam = 0.001
        # self._w = la.inv(g.transpose().dot(g) + lam * np.identity(len(centers))).dot(g.transpose()).dot(
        #     self._y_star_test)
        self._y_test = g.dot(self._w)

        accuracy = 1 - np.sum(
            np.sign(np.abs(np.argmax(self._y_test, axis=1) - np.argmax(self._y_star_test, axis=1)))) / len(
            self._data_test)
        print(f'accuracy test: {accuracy}')
        self._y_test = np.argmax(self._y_test, axis=1)

        # # plot circle:
        # ax = plt.gca()
        # cc = plt.Circle(centers, radius_vectors, facecolor='none', edgecolor='black')
        # ax.add_patch(cc)
        # plt.axis('scaled')
        # plt.show()
