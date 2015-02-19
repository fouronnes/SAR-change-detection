import numpy as np
from plotting import *

root = "../SAR_Data"

def read_sar_file(path, dtype, header):
    "Load a single SAR file"
    # Load data and reshape
    array = np.fromfile(path, dtype=dtype)
    if header:
        array = array[1024:]
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

class Region(object):
    "Defines a rectangular area in an image"
    def __init__(self, range_i, range_j):
        self.range_i = range_i
        self.range_j = range_j

class SARData(object):
    """
    Object representing a polarimetric SAR image
    using covariance matrix representation
    """

    def load(self, code, shape, header):
        "Load SARData object for a given month code"
        self.shape = shape
        self.size = shape[0]*shape[1]
        extension = ".emi" if header else ""
        self.hhhh = read_sar_file(root + '/{}/{}hhhh{}'.format(code, code, extension), np.float32, header)
        self.hhhv = read_sar_file(root + '/{}/{}hhhv{}'.format(code, code, extension), np.complex64, header)
        self.hvhv = read_sar_file(root + '/{}/{}hvhv{}'.format(code, code, extension), np.float32, header)
        self.hhvv = read_sar_file(root + '/{}/{}hhvv{}'.format(code, code, extension), np.complex64, header)
        self.hvvv = read_sar_file(root + '/{}/{}hvvv{}'.format(code, code, extension), np.complex64, header)
        self.vvvv = read_sar_file(root + '/{}/{}vvvv{}'.format(code, code, extension), np.float32, header)
        return self

    def region(self, region):
        "Extract a subset of the SARData image defined by a Region object"
        s = SARData()
        s.hhhh = self.hhhh.reshape(self.shape)[np.ix_(region.range_i, region.range_j)].flatten()
        s.hhhv = self.hhhv.reshape(self.shape)[np.ix_(region.range_i, region.range_j)].flatten()
        s.hvhv = self.hvhv.reshape(self.shape)[np.ix_(region.range_i, region.range_j)].flatten()
        s.hhvv = self.hhvv.reshape(self.shape)[np.ix_(region.range_i, region.range_j)].flatten()
        s.hvvv = self.hvvv.reshape(self.shape)[np.ix_(region.range_i, region.range_j)].flatten()
        s.vvvv = self.vvvv.reshape(self.shape)[np.ix_(region.range_i, region.range_j)].flatten()
        return s

    def masked_region(self, mask):
        "Extract a subset of the SARData image defined by a mask"
        pass

    def determinant(self):
        "Determinants of the covariance matrices in a SARData object"
        return np.real((self.hhhh*self.hvhv*self.vvvv
            + self.hhhv*self.hvvv*np.conj(self.hhvv)
            + self.hhvv*np.conj(self.hhhv)*np.conj(self.hvvv)
            - self.hhvv*self.hvhv*np.conj(self.hhvv)
            - self.hhhv*np.conj(self.hhhv)*self.vvvv
            - self.hhhh*self.hvvv*np.conj(self.hvvv)))


print("Loading SAR data...")

# Define notable regions in the SAR data set
region_complete = Region(range(0, 1024), range(0, 1024))
region_nochange = Region(range(307, 455), range(52, 120))
region_rye = Region(range(116, 146), range(328, 411))
region_grass = Region(range(268, 330), range(128, 234))

# Load data
march  = SARData().load("fl062_l", (1024, 1024), header=True)
april  = SARData().load("fl063_l", (1024, 1024), header=False)
may    = SARData().load("fl064_l", (1024, 1024), header=False)
june   = SARData().load("fl065_l", (1024, 1024), header=False)
july   = SARData().load("fl068_l", (1024, 1024), header=False)
august = SARData().load("fl074_l", (1024, 1024), header=True)

# The complete time series
sar_list = [march, april, may, june, july, august]

# Time series of image regions
sar_list_nochange = [X.region(region_nochange) for X in sar_list]
sar_list_rye      = [X.region(region_rye)      for X in sar_list]
sar_list_grass    = [X.region(region_grass)    for X in sar_list]

# No change region
# This is redundant with sar_list_nochange, but gamma.py uses this for now
march_no_change  = march.region(region_nochange)
april_no_change  = april.region(region_nochange)
may_no_change    = may.region(region_nochange)
june_no_change   = june.region(region_nochange)
july_no_change   = july.region(region_nochange)
august_no_change = august.region(region_nochange)

# Make color composites
plt.imsave("fig/march.jpg", color_composite(march))
plt.imsave("fig/april.jpg", color_composite(april))
plt.imsave("fig/may.jpg", color_composite(may))
plt.imsave("fig/june.jpg", color_composite(june))
plt.imsave("fig/july.jpg", color_composite(july))
plt.imsave("fig/august.jpg", color_composite(august))

