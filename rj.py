import numpy as np
from numpy import log
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *
from omnibus import Omnibus

def rj_test_statistic(sar_list, n, j):
    p = 3
    f = p**2
    # Note: sar_list (list of months) indexes are zero-based
    # but mathematical is 1 based, so indexes are off by -1
    # additionally ranges end points are off by +1 because exclusive, and recall -1+1 = 0
    sum_j = determinant(sar_sum(sar_list[:j]))
    sum_j_minus_1 = determinant(sar_sum(sar_list[:j-1]))

    detXj = determinant(sar_list[j-1]) # -1 because sar_list if 0-based
    lnR = n*(p*(j*log(j) - (j-1)*log(j-1)) + (j-1)*log(sum_j_minus_1) + log(detXj) - j*log(sum_j))

    return lnR

class RjTest(object):
    """
    Implements the 'Rj' test statistic
    Detects the point of change in time series of polarimetric SAR
    """
    def __init__(self, sar_list, ENL):
        """
        Create a new Rj test
        sar_list should be a list of SARData objects
        ENL is the (common) equivalent number of looks of the images
        """

        self.sar_list = sar_list
        self.ENL = ENL

        p = 3
        k = len(sar_list)
        self.k = k
        self.f = p**2
        n = ENL

        # Hypothesis H[l] is
        # Sl = ... = Sk
        self.H = np.zeros(k)
        self.H[:] = np.nan

        # Hypothesis K[l, s] is
        # S(l+s) = S(l+s-1)
        self.K = np.zeros((k, k, sar_list[0].hhhh.shape[0], sar_list[0].hhhh.shape[1]))
        nan_array = np.empty((sar_list[0].hhhh.shape[0], sar_list[0].hhhh.shape[1]))
        nan_array[:,:] = np.nan
        self.K[:,:] = nan_array

        for l in range(1, k):
            self.H[l] = Omnibus(sar_list[l-1:], ENL).pvalue()

            for j in range(1, k-l+1):
                # l-1 because sar_list indexing is 0-based
                # j+1 because K[l,j] is a test for S(j+1) = S(j),
                # while R(j) is a test for S(j) = S(j-1)
                self.K[l, j] = rj_test_statistic(sar_list[l-1:], ENL, j+1)

    def pvalue(self, l, j):
        "Average probability over the region"
        return 1 - np.mean(scipy.stats.chi2.cdf( -2 * self.K[l, j], df=self.f))

    def points_of_change(self, percent):
        """
        Index of change time point above significance level
        Returns list of (1-based) indexes such that each index is the first month of each change

        Example 1:
        M1 = M2 \= M3 = M4 = M5
        Returns: [2]

        Example 2:
        M1 \= M2 = M3 = M4 \= M5
        Returns: [1, 4]

        Example 3:
        M1 = M2 = M3 = M4 = M5
        Returns: []
        """

        result = []
        j = 1
        l = 1
        while j < self.k - l + 1:
            if self.pvalue(l, j) < percent:
                result.append((l + j - 1, self.pvalue(l, j)))
                l += j
                j = 1
            else:
                j += 1
        return result

if __name__ == "__main__":

    def print_pvalue_table(rj):
        "Pretty-print the table of p-values"
        print("""
        Apr = Mar | {:6.4f} 
        May = Apr | {:6.4f} {:6.4f} 
        Jun = May | {:6.4f} {:6.4f} {:6.4f} 
        Jul = Jun | {:6.4f} {:6.4f} {:6.4f} {:6.4f} 
        Aug = Jul | {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.4f}
        ----------|
        P(Q < q)  | {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.4f}
        """.format(
            rj.pvalue(1, 1),
            rj.pvalue(1, 2), rj.pvalue(2, 1),
            rj.pvalue(1, 3), rj.pvalue(2, 2), rj.pvalue(3, 1),
            rj.pvalue(1, 4), rj.pvalue(2, 3), rj.pvalue(3, 2), rj.pvalue(4, 1),
            rj.pvalue(1, 5), rj.pvalue(2, 4), rj.pvalue(3, 3), rj.pvalue(4, 2), rj.pvalue(5, 1),

            rj.H[1], rj.H[2], rj.H[3], rj.H[4], rj.H[5]
        ))

    print("Rj test...")

    # print("All:")
    # rj_all = RjTest(sar_list, 13)
    # print_pvalue_table(rj_all)
    # print(rj_all.points_of_change(0.05))

    print("")
    print("Forest:")
    rj_nochange = RjTest(sar_list_nochange, 13)
    print_pvalue_table(rj_nochange)
    print(rj_nochange.points_of_change(0.05))

    print("")
    print("Rye:")
    rj_rye = RjTest(sar_list_rye, 13)
    print_pvalue_table(rj_rye)
    print(rj_rye.points_of_change(0.05))

    print("")
    print("Grass:")
    rj_grass = RjTest(sar_list_grass, 13)
    print_pvalue_table(rj_grass)
    print(rj_grass.points_of_change(0.05))


