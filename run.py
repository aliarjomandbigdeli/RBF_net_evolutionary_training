import matplotlib.pyplot as plt
import rbf_es

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""


def main():
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


if __name__ == '__main__':
    main()
