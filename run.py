import matplotlib.pyplot as plt
import rbf_es

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""


def main():
    sample_num = 100
    iter_num = 60
    my_rbf_reg = rbf_es.RBFRegression()
    my_rbf_reg.create_random_dataset(sample_num, 1)
    my_rbf_reg.train(iter_num, my_rbf_reg.data())

    # -- unsorted
    # plt.scatter(my_rbf_reg.data(), my_rbf_reg.y(), label='RBF-Net')
    # plt.scatter(my_rbf_reg.data(), my_rbf_reg.y_star(), '-o', label='true')
    # -- unsorted

    # -- sorted samples(usually is in regression)
    plt.plot(my_rbf_reg.data(), my_rbf_reg.y(), '-o', label='RBF-Net')
    plt.plot(my_rbf_reg.data(), my_rbf_reg.y_star(), '-o', label='true')
    plt.tight_layout()
    # -- sorted
    plt.legend()
    plt.title(f'number of samples: {sample_num} ,number of iterations:  {iter_num}')
    plt.show()


if __name__ == '__main__':
    main()
