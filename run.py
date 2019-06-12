import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

import rbf_es

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""

colors = {-2: 'black', - 1: 'red', 0: 'red', 1: 'blue', 2: 'green', 3: 'cyan', 4: 'yellow', 5: 'black', 6: 'magenta'}


def main():
    print('RBF Network')

    # run_regression()
    run_bin_classification()
    # run_classification()


def run_regression():
    # random data:
    # for i in range(4):
    #     sample_num = 100
    #     iter_num = 20
    #     my_rbf_reg = rbf_es.RBFRegression()
    #     my_rbf_reg.create_random_dataset(sample_num, 1)
    #     my_rbf_reg.initialize_parameters_based_on_data()
    #     my_rbf_reg.train(iter_num, my_rbf_reg.data())
    #
    #     plt.figure()
    #     # -- unsorted
    #     # plt.scatter(my_rbf_reg.data(), my_rbf_reg.y(), label='RBF-Net')
    #     # plt.scatter(my_rbf_reg.data(), my_rbf_reg.y_star(), '-o', label='true')
    #     # -- unsorted
    #
    #     # -- sorted samples(usually is in regression)
    #     plt.plot(my_rbf_reg.data(), my_rbf_reg.y(), '-o', label='RBF-Net')
    #     plt.plot(my_rbf_reg.data(), my_rbf_reg.y_star(), '-o', label='true')
    #     # -- sorted
    #     plt.legend()
    #     plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
    #     # plt.show()
    #     plt.savefig(f'{i}-1.png')
    #
    #     plt.figure()
    #     plt.plot(my_rbf_reg._best_fitness_list, '-o', label='best')
    #     plt.plot(my_rbf_reg._avg_fitness_list, '-o', label='average')
    #     plt.legend()
    #     plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
    #     plt.savefig(f'{i}-2.png')

    # read excel
    iter_num = 10
    my_rbf_reg = rbf_es.RBFRegression()
    my_rbf_reg.read_excel("regdata2000.xlsx")
    my_rbf_reg.initialize_parameters_based_on_data()
    my_rbf_reg.train(iter_num, my_rbf_reg.data())

    plt.figure()

    # -- sorted samples(usually is in regression)
    plt.plot(my_rbf_reg.y(), '-o', label='RBF-Net')
    plt.plot(my_rbf_reg.y_star(), '-o', label='true')
    # -- sorted
    plt.legend()
    plt.title(f'number of samples: 2000 ,number of iterations:  {iter_num}')
    # plt.show()
    plt.savefig(f'result.png')


def run_bin_classification():
    for i in range(1):
        sample_num = 5000
        iter_num = 5
        my_rbf_bin = rbf_es.RBFBinClassifier()
        my_rbf_bin.create_random_dataset(sample_num, 2, 2)
        # my_rbf_bin.read_excel("train.xlsx")
        my_rbf_bin.read_excel("2clstrain5000.xlsx")
        my_rbf_bin.initialize_parameters_based_on_data()
        my_rbf_bin.train(iter_num, my_rbf_bin.data())

        df = DataFrame(dict(x=my_rbf_bin.data()[:, 0], y=my_rbf_bin.data()[:, 1], label=my_rbf_bin._y_star))
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, figure index: {i}')
        plt.savefig(f'{i}-1.png')

        # print(my_rbf_bin.data()[:, 0])
        # print(my_rbf_bin.data()[:, 1])
        # print(my_rbf_bin._y_star)
        # print(f'shape: {my_rbf_bin._y_star.shape}')
        # print(my_rbf_bin.y())
        # print(np.around(my_rbf_bin.y()))
        accuracy = 1 - np.sum(np.abs(0.5 * np.around(my_rbf_bin.y()) + 0.5 - np.around(my_rbf_bin.y()))) / len(
            np.around(my_rbf_bin.y()))
        print(f'accuracy: {accuracy}')
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


def run_classification():
    sample_num = 300
    iter_num = 20
    my_rbf_classifier = rbf_es.RBFClassifier()
    # my_rbf_classifier.create_random_dataset(sample_num, 3, 2)
    my_rbf_classifier.read_excel("5clstest5000.xlsx")
    my_rbf_classifier.initialize_parameters_based_on_data()
    my_rbf_classifier.train(iter_num, my_rbf_classifier.data())

    df = DataFrame(dict(x=my_rbf_classifier.data()[:, 0], y=my_rbf_classifier.data()[:, 1],
                        label=my_rbf_classifier._y_star_before_1hot))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}')
    plt.savefig(f'1.png')

    # print(my_rbf_classifier.data()[:, 0])
    # print(my_rbf_classifier.data()[:, 1])
    print(my_rbf_classifier._y_star)
    print(my_rbf_classifier._y_star_before_1hot)
    # print(f'shape: {my_rbf_classifier._y_star.shape}')
    # print(my_rbf_classifier.y())
    # print(np.around(my_rbf_classifier.y()))

    plt.figure()
    df = DataFrame(dict(x=my_rbf_classifier.data()[:, 0], y=my_rbf_classifier.data()[:, 1],
                        label=np.around(my_rbf_classifier.y())))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    plt.legend()
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}')
    plt.savefig(f'2.png')

    plt.figure()
    plt.plot(my_rbf_classifier._best_fitness_list, '-o', label='best')
    plt.plot(my_rbf_classifier._avg_fitness_list, '-o', label='average')
    plt.legend()
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}')
    plt.savefig(f'3.png')


if __name__ == '__main__':
    main()
