import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

import rbf_es

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""

colors = {-2: 'black', - 1: 'orange', 0: 'red', 1: 'blue', 2: 'green', 3: 'cyan', 4: 'yellow', 5: 'black', 6: 'magenta'}


def main():
    print('RBF Network')

    run_regression()
    run_bin_classification()
    run_classification()


def run_classification():
    sample_num = 1500
    test_size = 5000
    iter_num = 5
    my_rbf_classifier = rbf_es.RBFClassifier()
    my_rbf_classifier.read_excel("dataset/5clstrain1500.xlsx", "dataset/5clstest5000.xlsx")
    my_rbf_classifier.initialize_parameters_based_on_data()
    my_rbf_classifier.train(iter_num, my_rbf_classifier.data())
    my_rbf_classifier.test()

    df = DataFrame(dict(x=my_rbf_classifier.data()[:, 0], y=my_rbf_classifier.data()[:, 1],
                        label=my_rbf_classifier._y_star_before_1hot))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, training data')
    plt.savefig(f'1.png')

    plt.figure()
    df = DataFrame(dict(x=my_rbf_classifier.data()[:, 0], y=my_rbf_classifier.data()[:, 1],
                        label=my_rbf_classifier.y()))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.legend()
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, trained result')
    plt.savefig(f'2.png')

    plt.figure()
    df = DataFrame(
        dict(x=my_rbf_classifier._data_test[:, 0], y=my_rbf_classifier._data_test[:, 1],
             label=my_rbf_classifier._y_star_test_before_1hot))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    plt.legend()
    plt.title(f'number of samples: {test_size} ,number of iterations:  {iter_num}, test data')
    plt.savefig(f'3.png')

    plt.figure()
    df = DataFrame(
        dict(x=my_rbf_classifier._data_test[:, 0], y=my_rbf_classifier._data_test[:, 1],
             label=np.around(my_rbf_classifier._y_test)))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    plt.legend()
    plt.title(f'number of samples: {test_size} ,number of iterations:  {iter_num}, test predict')
    plt.savefig(f'4.png')


def run_regression():
    # read excel
    iter_num = 6
    my_rbf_reg = rbf_es.RBFRegression()
    my_rbf_reg.read_excel("dataset/regdata2000.xlsx")
    my_rbf_reg.initialize_parameters_based_on_data()
    my_rbf_reg.train(iter_num, my_rbf_reg.data())
    my_rbf_reg.test()

    plt.figure()
    plt.plot(my_rbf_reg._y_test, '-o', label='RBF-Net')
    plt.plot(my_rbf_reg._y_star_test, '-', label='true')
    # -- sorted
    plt.legend()
    plt.title(f'number of samples: 2000 ,number of iterations:  {iter_num}')
    # plt.show()
    plt.savefig(f'reg-result.png')


def run_bin_classification():
    sample_num = 1500
    test_num = 5000
    iter_num = 5
    my_rbf_bin = rbf_es.RBFBinClassifier()
    my_rbf_bin.create_random_dataset(sample_num, 2, 2)
    my_rbf_bin.read_excel("dataset/2clstrain1500.xlsx", "dataset/2clstest5000.xlsx")
    my_rbf_bin.initialize_parameters_based_on_data()
    my_rbf_bin.train(iter_num, my_rbf_bin.data())
    my_rbf_bin.test()

    df = DataFrame(dict(x=my_rbf_bin.data()[:, 0], y=my_rbf_bin.data()[:, 1], label=my_rbf_bin._y_star))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, training data')
    plt.savefig(f'bin-1.png')

    plt.figure()
    df = DataFrame(dict(x=my_rbf_bin.data()[:, 0], y=my_rbf_bin.data()[:, 1], label=np.sign(my_rbf_bin.y())))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    plt.legend()
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}, trained result')
    plt.savefig(f'bin-2.png')

    plt.figure()
    df = DataFrame(
        dict(x=my_rbf_bin._data_test[:, 0], y=my_rbf_bin._data_test[:, 1], label=my_rbf_bin._y_star_test))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    plt.legend()
    plt.title(f'number of samples: {test_num} ,number of iterations:  {iter_num}, test data, test data')
    plt.savefig(f'bin-3.png')

    plt.figure()
    df = DataFrame(
    dict(x=my_rbf_bin._data_test[:, 0], y=my_rbf_bin._data_test[:, 1], label=np.sign(my_rbf_bin._y_test)))
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # plt.show()
    plt.legend()
    plt.title(
        f'number of samples: {test_num} ,number of iterations:  {iter_num}, test predict, test predict')
    plt.savefig(f'bin-4.png')

    # plt.figure()
    # plt.plot(my_rbf_bin._best_fitness_list, '-o', label='best')
    # plt.plot(my_rbf_bin._avg_fitness_list, '-o', label='average')
    # plt.legend()
    # plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}')
    # plt.savefig(f'bin-5.png')


if __name__ == '__main__':
    main()
