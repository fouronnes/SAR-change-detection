import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import determinant, sar_sum

def compute_omnibus(sar_list, n):
    """Compute (ln Q) in the Omnibus test"""


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

        self.p = 3
        self.k = len(sar_list)

        X = sar_sum(sar_list)
        sum_term = np.sum([np.log(determinant(X)) for X in sar_list])

        # Omnibus test
        self.lnq = n*(p*k* np.log(self.k) + sum_term - self.k*np.log(determinant(X)))

    def image_binary(self, percent):
        # Select threshold from chi2 percentile (ignore w2 term)
        p = 3
        chi2 = scipy.stats.chi2(p**2)
        threshold = chi2.ppf(1.0 - percent)

        im = np.zeros_like(self.lnq)
        im[-2*self.lnq > threshold] = 1
        return im

    def image_linear(self, p1, p2):
        pass

