import numpy
import sys

home_dir = '../'
sys.path.append(home_dir)


def load_data(data_name, file_path=None):
    file_name = home_dir + 'Resources/' + data_name + '.npz'
    if file_path is not None:
        file_name = file_path
    npz_file = numpy.load(file_name)

    x_mat = npz_file['X']
    y_vec = npz_file['y']
    n, d = x_mat.shape

    print(npz_file.files)
    print('size of X is ' + str(n) + '-by-' + str(d))
    print('size of y is ' + str(y_vec.shape))

    return x_mat, y_vec
