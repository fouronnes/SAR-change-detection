import numpy as np

root = "../SAR_Data"

def read_sar_file(path, dtype):
    "Load a single SAR file"
    # Load data and reshape
    array = np.fromfile(path, dtype=dtype)
    array = array.reshape((1024, 1024))
    # Swap byte order
    return array.newbyteorder('S')

def sar_sum(sar_list):
    "Sum of a list of SARData covariance matrices objects"
    s = SARData()
    s.hhhh = np.sum([X.hhhh for X in sar_list])
    s.hhhv = np.sum([X.hhhv for X in sar_list])
    s.hvhv = np.sum([X.hvhv for X in sar_list])
    s.hhvv = np.sum([X.hhvv for X in sar_list])
    s.hvvv = np.sum([X.hvvv for X in sar_list])
    s.vvvv = np.sum([X.vvvv for X in sar_list])
    return s

def determinant(X):
    "Determinants of the covariance matrices in a SARData object"
    detX = np.empty((1024, 1024))
    return np.real((X.hhhh*X.hvhv*X.vvvv
         + X.hhhv*X.hvvv*np.conj(X.hhvv)
         + X.hhvv*np.conj(X.hhhv)*np.conj(X.hvvv)
         - X.hhvv*X.hvhv*np.conj(X.hhvv)
         - X.hhhv*np.conj(X.hhhv)*X.vvvv
         - X.hhhh*X.hvvv*np.conj(X.hvvv)))

def region(X, range_i, range_j):
    "Extract sub-region of a SARData image"
    s = SARData()
    s.hhhh = X.hhhh[np.ix_(range_i, range_j)]
    s.hhhv = X.hhhv[np.ix_(range_i, range_j)]
    s.hvhv = X.hvhv[np.ix_(range_i, range_j)]
    s.hhvv = X.hhvv[np.ix_(range_i, range_j)]
    s.hvvv = X.hvvv[np.ix_(range_i, range_j)]
    s.vvvv = X.vvvv[np.ix_(range_i, range_j)]
    return s

class SARData(object):
    """
    Object representing a polarimetric SAR image
    using covariance matrix representation
    """

    def load_april(self):
        "Load SARData object for April dataset"
        self.hhhh = read_sar_file(root + '/fl063_l/fl063_lhhhh', np.float32)
        self.hhhv = read_sar_file(root + '/fl063_l/fl063_lhhhv', np.complex64)
        self.hvhv = read_sar_file(root + '/fl063_l/fl063_lhvhv', np.float32)
        self.hhvv = read_sar_file(root + '/fl063_l/fl063_lhhvv', np.complex64)
        self.hvvv = read_sar_file(root + '/fl063_l/fl063_lhvvv', np.complex64)
        self.vvvv = read_sar_file(root + '/fl063_l/fl063_lvvvv', np.float32)
        return self

    def load_may(self):
        "Load SARData object for May dataset"
        self.hhhh = read_sar_file(root + '/fl064_l/fl064_lhhhh', np.float32)
        self.hhhv = read_sar_file(root + '/fl064_l/fl064_lhhhv', np.complex64)
        self.hvhv = read_sar_file(root + '/fl064_l/fl064_lhvhv', np.float32)
        self.hhvv = read_sar_file(root + '/fl064_l/fl064_lhhvv', np.complex64)
        self.hvvv = read_sar_file(root + '/fl064_l/fl064_lhvvv', np.complex64)
        self.vvvv = read_sar_file(root + '/fl064_l/fl064_lvvvv', np.float32)
        return self

