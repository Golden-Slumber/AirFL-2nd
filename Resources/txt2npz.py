import numpy
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import sys

home_dir = '../'
sys.path.append(home_dir)


def txt2npz(data_name, is_transform_y=False, is_remove_empty=False):
    filename = home_dir + 'Resource/' + data_name
    x_mat, y_vec = load_svmlight_file(filename)
    x_mat = numpy.array(x_mat.todense())
    n, d = x_mat.shape
    print('size of X is ' + str(n) + '-by-' + str(d))
    print('size of y is ' + str(y_vec.shape))
    x_mat = MinMaxScaler().fit_transform(x_mat)

    if is_transform_y:
        # for cross entropy or logistic regression
        y_vec = (y_vec * 2) - 3

    if is_remove_empty:
        sum_x = numpy.sum(numpy.abs(x_mat), axis=1)
        idx = (sum_x > 1e-6)
        x_mat = x_mat[idx, :]
        y_vec = y_vec[idx]

    out_filename = home_dir + 'Resource/' + data_name + '.npz'
    numpy.savez(out_filename, data_name=data_name, x_mat=x_mat, y_vec=y_vec)


if __name__ == '__main__':
    data_name = 'covtype'
    txt2npz(data_name, is_transform_y=False)
