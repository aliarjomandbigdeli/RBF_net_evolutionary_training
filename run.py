import matplotlib.pyplot as plt
import rbf_es

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 6/2/2019
"""


def main():
    my_rbf_reg = rbf_es.RBFRegression()
    my_rbf_reg.create_random_dataset(100, 1)
    my_rbf_reg.train(100, my_rbf_reg.data())

    plt.scatter(my_rbf_reg.data(), my_rbf_reg.y())
    plt.show()

    plt.scatter(my_rbf_reg.data(), my_rbf_reg.y_star())
    plt.show()


if __name__ == '__main__':
    main()
