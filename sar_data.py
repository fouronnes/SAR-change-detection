import numpy as np
from plotting import *

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
    s.hhhh = sum([X.hhhh for X in sar_list])
    s.hhhv = sum([X.hhhv for X in sar_list])
    s.hvhv = sum([X.hvhv for X in sar_list])
    s.hhvv = sum([X.hhvv for X in sar_list])
    s.hvvv = sum([X.hvvv for X in sar_list])
    s.vvvv = sum([X.vvvv for X in sar_list])
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

    def load(self, code):
        "Load SARData object for a given month code"
        self.hhhh = read_sar_file(root + '/{}/{}hhhh'.format(code, code), np.float32)
        self.hhhv = read_sar_file(root + '/{}/{}hhhv'.format(code, code), np.complex64)
        self.hvhv = read_sar_file(root + '/{}/{}hvhv'.format(code, code), np.float32)
        self.hhvv = read_sar_file(root + '/{}/{}hhvv'.format(code, code), np.complex64)
        self.hvvv = read_sar_file(root + '/{}/{}hvvv'.format(code, code), np.complex64)
        self.vvvv = read_sar_file(root + '/{}/{}vvvv'.format(code, code), np.float32)
        return self

print("Loading SAR data...")

# Ranges defining the forest region of the image
no_change_i = range(307, 455)
no_change_j = range(52, 120)

# Load data
# march = SARData().load("fl062_l")
april = SARData().load("fl063_l")
may = SARData().load("fl064_l")
june = SARData().load("fl065_l")
july = SARData().load("fl068_l")
# august = SARData().load("fl074_l")

# No change region
april_no_change = region(april, no_change_i, no_change_j)
may_no_change = region(may, no_change_i, no_change_j)
june_no_change = region(june, no_change_i, no_change_j)
july_no_change = region(july, no_change_i, no_change_j)

# Make color composites
plt.imsave("fig/april.jpg", color_composite(april))
plt.imsave("fig/may.jpg", color_composite(may))

