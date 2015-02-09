import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *

class Omnibus(object):
    """
    Implements the Omnibus test statistic
    """
    def __init__(self, sar_list, ENL):
        """
        Create a new Omnibus test
        sar_list should be a list of SARData objects
        ENL is the (common) equivalent number of looks of the images
        """

        self.sar_list = sar_list
        self.ENL = ENL

        p = 3
        k = len(sar_list)
        n = ENL
        X = sar_sum(sar_list)
        sum_term = np.sum([np.log(determinant(X)) for X in sar_list])

        # Omnibus test
        self.lnq = n*(p*k* np.log(k) + sum_term - k*np.log(determinant(X)))

    def image_binary(self, percent):
        # Select threshold from chi2 percentile (ignore w2 term)
        p = 3
        k = len(self.sar_list)
        f = (k-1)*p**2
        chi2 = scipy.stats.chi2(f)
        threshold = chi2.ppf(1.0 - percent)

        im = np.zeros_like(self.lnq)
        im[-2*self.lnq > threshold] = 1
        return im

    def image_linear(self, p1, p2):
        pass

o = Omnibus([april, may, june, july], 13)

im = o.image_binary(0.10)
