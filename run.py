import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

import rbf_es

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""

colors = {0: 'red', 1: 'blue', 2: 'green'}


def main():
    print('RBF Network')

    # run_regression()
    run_bin_classification()


def run_regression():
    for i in range(4):
        sample_num = 100
        iter_num = 60
        my_rbf_reg = rbf_es.RBFRegression()
        my_rbf_reg.create_random_dataset(sample_num, 1)
        my_rbf_reg.initialize_parameters_based_on_data()
        my_rbf_reg.train(iter_num, my_rbf_reg.data())

        plt.figure()
        # -- unsorted
        # plt.scatter(my_rbf_reg.data(), my_rbf_reg.y(), label='RBF-Net')
        # plt.scatter(my_rbf_reg.data(), my_rbf_reg.y_star(), '-o', label='true')
        # -- unsorted

        # -- sorted samples(usually is in regression)
        plt.plot(my_rbf_reg.data(), my_rbf_reg.y(), '-o', label='RBF-Net')
        plt.plot(my_rbf_reg.data(), my_rbf_reg.y_star(), '-o', label='true')
        # -- sorted
        plt.legend()
        plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
        # plt.show()
        plt.savefig(f'{i}-1.png')

        plt.figure()
        plt.plot(my_rbf_reg._best_fitness_list, '-o', label='best')
        plt.plot(my_rbf_reg._avg_fitness_list, '-o', label='average')
        plt.legend()
        plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
        plt.savefig(f'{i}-2.png')


def run_bin_classification():
    for i in range(1):
        sample_num = 100
        iter_num = 30
        my_rbf_bin = rbf_es.RBFBinClassifier()
        my_rbf_bin.create_random_dataset(100, 2, 2)
        my_rbf_bin.initialize_parameters_based_on_data()
        my_rbf_bin.train(iter_num, my_rbf_bin.data())

        df = DataFrame(dict(x=my_rbf_bin.data()[:, 0], y=my_rbf_bin.data()[:, 1], label=my_rbf_bin._y_star))
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
        plt.savefig(f'{i}-1.png')

        print(my_rbf_bin._y_star)
        print(my_rbf_bin.y())
        print(np.around(my_rbf_bin.y()))

        plt.figure()
        df = DataFrame(dict(x=my_rbf_bin.data()[:, 0], y=my_rbf_bin.data()[:, 1], label=np.around(my_rbf_bin.y())))
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        # plt.show()
        plt.legend()
        plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
        plt.savefig(f'{i}-2.png')

        plt.figure()
        plt.plot(my_rbf_bin._best_fitness_list, '-o', label='best')
        plt.plot(my_rbf_bin._avg_fitness_list, '-o', label='average')
        plt.legend()
        plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
        plt.savefig(f'{i}-3.png')


if __name__ == '__main__':
    main()
