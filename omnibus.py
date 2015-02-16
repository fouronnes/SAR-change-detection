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

        self.f = (k-1)*p**2
        self.rho = 1- (2*p**2 - 1)/(6*p*(k-1)) * (k/n - 1/(n*k))

        sum_term = sum([np.log(determinant(Xi)) for Xi in sar_list])
        X = sar_sum(sar_list)

        # Omnibus test
        self.lnq = n*(p*k*np.log(k) + sum_term - k*np.log(determinant(X)))

    def histogram(self):
        """
        Histogram of no change region
        and pdf with only chi2 term
        """

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(-2*self.lnq.flatten(), bins=100, normed=True, color="#3F5D7D")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Overlay pdf
        p = 3
        k = len(self.sar_list)
        f = (k-1)*p**2
        chi2 = scipy.stats.chi2(f)

        x = np.linspace(0, 100, 1000)
        y = chi2.pdf(x)
        ax.plot(x, y, color="black", linewidth=2)

        ax.set_xlim([0, 100])

        return fig, ax

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

    def image_autothresholds(self):
        pass

    def image_linear(self, p1, p2):
        pass

if __name__ == "__main__":
    print("Omnibus test...")

    # Omnibus test of the entire image
    omnibus = Omnibus(sar_list, 13)

    # Omnibus test over the no change region
    omnibus_no_change = Omnibus(sar_list_nochange, 13)

    # Histogram over the no change region
    fig, ax = omnibus_no_change.histogram()
    hist_filename = "fig/omnibus/hist.nochange.pdf"
    fig.savefig(hist_filename, bbox_inches='tight')

    # Histogram, entire region
    fig, ax = omnibus.histogram()
    hist_filename = "fig/omnibus/hist.total.pdf"
    fig.savefig(hist_filename, bbox_inches='tight')

    # Binary images
    def omnibus_binary(percent):
        im = omnibus.image_binary(percent)
        plt.imsave("fig/omnibus/omnibus.{}.jpg".format(percent), im, cmap="gray")

    omnibus_binary(0.00001)
    omnibus_binary(0.0001)
    omnibus_binary(0.001)
    omnibus_binary(0.01)
    omnibus_binary(0.05)
    omnibus_binary(0.10)

    def average_probability(omnibus):
        rho = omnibus.rho
        lnq = omnibus.lnq
        f = omnibus.f
        return 1 - np.mean(scipy.stats.chi2.cdf( -2 * rho * lnq, df=f))

    # Pairwise test
    def average_test_for_region(r):
        print("March = April :", average_probability(Omnibus([march.region(r), april.region(r)], 13)))
        print("April = May   :", average_probability(Omnibus([april.region(r), may.region(r)], 13)))
        print("May   = June  :", average_probability(Omnibus([may.region(r), june.region(r)], 13)))
        print("June  = July  :", average_probability(Omnibus([june.region(r), july.region(r)], 13)))
        print("July  = August:", average_probability(Omnibus([july.region(r), august.region(r)], 13)))
        print("Omnibus       :", average_probability(Omnibus([X.region(r) for X in sar_list], 13)))

    # Omnibus test in notable regions
    print("")
    print("Forest:")
    average_test_for_region(region_nochange)

    print("")
    print("Rye:")
    average_test_for_region(region_rye)

    print("")
    print("Grass:")
    average_test_for_region(region_grass)
