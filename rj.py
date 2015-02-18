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

    def average_pvalue(self, l, j):
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
            if self.average_pvalue(l, j) < percent:
                result.append((l + j - 1, self.average_pvalue(l, j)))
                l += j
                j = 1
            else:
                j += 1
        return result

    def number_of_changes(self, percent):
        """
        The 'point of change' algorithm for each pixel,
        vectorized using numpy array operations
        """

        image_shape = self.K[1, 1].shape
        j = np.ones(image_shape, dtype=int)
        l = np.ones(image_shape, dtype=int)
        result = np.zeros(image_shape, dtype=np.float32)
        for repeat in range(1, self.k):
            # Numpy array indexing black magic to obtain an image mask
            # Indicating if there is change at this (l, j) time point
            a, b = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing="ij")
            change_mask = (1 - scipy.stats.chi2.cdf(-2*self.K[l, j, a, b], df=self.f)) < percent

            # Where there is change
            l[change_mask] += j[change_mask]
            j[change_mask] = 1
            result[change_mask] += 1

            # Where there is no change
            j[np.logical_not(change_mask)] += 1

        return result

def number_of_changes_histogram(im):
    f = plt.figure(figsize=(4, 2))
    ax = f.add_subplot(111)
    ax.hist(im.flatten(), bins=np.arange(6), normed=True, rwidth=0.9, align="left", color="#3F5D7D")
    ax.grid(True, axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim([-0.5, 5.5])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    return f, ax

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
            rj.average_pvalue(1, 1),
            rj.average_pvalue(1, 2), rj.average_pvalue(2, 1),
            rj.average_pvalue(1, 3), rj.average_pvalue(2, 2), rj.average_pvalue(3, 1),
            rj.average_pvalue(1, 4), rj.average_pvalue(2, 3), rj.average_pvalue(3, 2), rj.average_pvalue(4, 1),
            rj.average_pvalue(1, 5), rj.average_pvalue(2, 4), rj.average_pvalue(3, 3), rj.average_pvalue(4, 2), rj.average_pvalue(5, 1),

            rj.H[1], rj.H[2], rj.H[3], rj.H[4], rj.H[5]
        ))

    def number_of_changes_test(rj, name, percent):
        """Produce an histogram and image of the total number of changes detected"""
        im = rj.number_of_changes(percent)
        plt.imsave("fig/rj/{}/number_of_changes.{}.jpg".format(name, percent), im, vmin=0, vmax=5, cmap="gray")
        f, ax = number_of_changes_histogram(im)
        f.savefig("fig/rj/{}/number_of_changes.hist.{}.pdf".format(name, percent), bbox_inches='tight')

    print("Rj test...")

    print("All:")
    rj_all = RjTest(sar_list, 13)
    print_pvalue_table(rj_all)
    number_of_changes_test(rj_all, "all", 0.10)
    number_of_changes_test(rj_all, "all", 0.05)
    number_of_changes_test(rj_all, "all", 0.01)
    number_of_changes_test(rj_all, "all", 0.001)
    number_of_changes_test(rj_all, "all", 0.0001)
    number_of_changes_test(rj_all, "all", 0.00001)

    print("")
    print("Forest:")
    rj_nochange = RjTest(sar_list_nochange, 13)
    print_pvalue_table(rj_nochange)
    number_of_changes_test(rj_nochange, "forest", 0.10)
    number_of_changes_test(rj_nochange, "forest", 0.05)
    number_of_changes_test(rj_nochange, "forest", 0.01)
    number_of_changes_test(rj_nochange, "forest", 0.001)
    number_of_changes_test(rj_nochange, "forest", 0.0001)
    number_of_changes_test(rj_nochange, "forest", 0.00001)

    print("")
    print("Rye:")
    rj_rye = RjTest(sar_list_rye, 13)
    print_pvalue_table(rj_rye)
    number_of_changes_test(rj_rye, "rye", 0.10)
    number_of_changes_test(rj_rye, "rye", 0.05)
    number_of_changes_test(rj_rye, "rye", 0.01)
    number_of_changes_test(rj_rye, "rye", 0.001)
    number_of_changes_test(rj_rye, "rye", 0.0001)
    number_of_changes_test(rj_rye, "rye", 0.00001)

    print("")
    print("Grass:")
    rj_grass = RjTest(sar_list_grass, 13)
    print_pvalue_table(rj_grass)
    number_of_changes_test(rj_grass, "grass", 0.10)
    number_of_changes_test(rj_grass, "grass", 0.05)
    number_of_changes_test(rj_grass, "grass", 0.01)
    number_of_changes_test(rj_grass, "grass", 0.001)
    number_of_changes_test(rj_grass, "grass", 0.0001)
    number_of_changes_test(rj_grass, "grass", 0.00001)


