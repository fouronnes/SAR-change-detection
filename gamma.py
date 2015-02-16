import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *

class Gamma(object):
    """
    Test statistic on equality of two Gamma parameters
    Use for change detection between two single channel SAR images X and Y
    n, m are Equivalent Number of Looks (ENL)
    """

    def __init__(self, X, Y, n, m):
        self.X = X
        self.Y = Y
        self.n = n
        self.m = m
        self.Q = Y/X # Test statistic

    def histogram(self, percentile):
        f = plt.figure(figsize=(8, 4))
        ax = f.add_subplot(111)
        ax.hist(self.Q.flatten(), bins=100, normed=True, range=(0,5), color='#3F5D7D')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        ax.set_xlabel('Test statistic')
        ax.set_ylabel('Frequency')

        ax.set_ylim([0, 1.1])

        # Fisher's F overlay
        F = scipy.stats.f(2*self.m, 2*self.n)
        x = np.linspace(0, 5, 500)
        ax.plot(x, F.pdf(x), color='black', linewidth=2)

        # Select threshold from distrib quantile
        t_inf, t_sup = F.ppf(percentile/2), F.ppf(1 - percentile/2)

        return f, ax
    
    def image_binary(self, percentile):
        F = scipy.stats.f(2*self.m, 2*self.n)
        t_inf, t_sup = F.ppf(percentile/2), F.ppf(1 - percentile/2)

        im = np.zeros_like(self.Q)
        im[self.Q < t_inf] = 1
        im[self.Q > t_sup] = 1

        return im

    def image_color2(self, percentile):
        """
        Change detection image with two colors indicating the change direction
        Black - Gray - White
        """
        F = scipy.stats.f(2*self.m, 2*self.n)
        t_inf, t_sup = F.ppf(percentile/2), F.ppf(1 - percentile/2)

        im = np.empty_like(self.Q)
        im[:,:] = 0.5
        im[self.Q < t_inf] = 0
        im[self.Q > t_sup] = 1

        return im

    def image_color3(self, percentile):
        """
        Change detection image with blue/red indicating the change direction
        """
        F = scipy.stats.f(2*self.m, 2*self.n)
        t_inf, t_sup = F.ppf(percentile/2), F.ppf(1 - percentile/2)

        im = np.empty((self.Q.shape[0], self.Q.shape[1], 3))
        im[:,:] = np.array([0, 0, 0])
        im[self.Q < t_inf] = np.array([170, 63, 57])
        im[self.Q > t_sup] = np.array([35, 100, 103])

        return im

    def image_linear(self, percentile):
        pass

def aregion(X):
    return X[np.ix_(region_nochange.range_i, region_nochange.range_j)]

def multiENL_gamma(april, may):
    gamma = Gamma(april, may, 13, 13)

    f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(111)
    ax.hist(gamma.Q.flatten(), bins=100, normed=True, range=(0,3), color='#3F5D7D')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlabel('Test statistic')
    ax.set_ylabel('Frequency')

    ax.set_ylim([0, 1.3])
    ax.set_xticks([0, 1, 2, 3])

    # Fisher's F overlay
    def overlay(ENL, side="right"):
        F = scipy.stats.f(2*ENL, 2*ENL)
        x = np.linspace(0, 3, 500)
        ax.plot(x, F.pdf(x), color='black', linewidth=1)
        mode = (ENL - 1)/(ENL+1)
        xtext = 0.4 if side == "left" else 1.3
        ax.annotate('{}'.format(ENL), xy=(mode, F.pdf(mode)), xytext=(xtext, F.pdf(mode)),
            arrowprops=dict(facecolor='black', shrink=0.05, width=.5, headwidth=2),
            fontsize=11,
            horizontalalignment='right',
            verticalalignment='center'
            )

    # overlay(8)
    overlay(9, "left")
    # overlay(10)
    overlay(11, "left")
    # overlay(12)
    overlay(13, "left")
    # overlay(14)
    overlay(15, "left")
    # overlay(16)
    overlay(17, "left")

    return f, ax

def critical_region():
    "Critical region figure"

    percent = 0.10

    f = plt.figure(figsize=(8, 3))
    ax = f.add_subplot(111)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlabel('Test statistic')
    ax.set_ylabel('Frequency')

    ax.set_ylim([0, 1.1])

    # Fisher's F pdf
    ENL = 13
    F = scipy.stats.f(2*ENL, 2*ENL)
    x = np.linspace(0, 3, 500)
    y = F.pdf(x)
    ax.plot(x, y, color='black', linewidth=1)
    
    # Thresholds
    t_inf, t_sup = F.ppf(percent/2), F.ppf(1 - percent/2)
    ax.fill_between(x, y, where=(x < t_sup) == (x > t_inf), color='#3F5D7D')

    ax.set_xticks([t_inf, t_sup])
    ax.set_xticklabels([r"$c_1$", r"$c_2$"], size=16)

    anotx = (ENL - 1)/(ENL+1) + 0.2
    ax.annotate(r'$F(2m, 2n)$', xy=(anotx, F.pdf(anotx)), xytext=(anotx + 0.6, F.pdf(anotx)),
        arrowprops=dict(facecolor='black', shrink=0.05, width=.3, headwidth=5),
        fontsize=16,
        horizontalalignment='right',
        verticalalignment='center'
        )

    ax.text(0.68, 0.5, "No change", color="white", size=16)
    ax.text(0.07, 0.5, "Change", color="black", size=16)
    ax.text(2.0, 0.5, "Change", color="black", size=16)

    ax.axvline(t_inf, color="black", linestyle="--")
    ax.axvline(t_sup, color="black", linestyle="--")

    return f, ax

if __name__ == "__main__":
    print("Gamma test...")

    def gamma_test(april, may, channel, ENL, percent):
        # Data
        X = april.__dict__[channel]
        Y = may.__dict__[channel]
        Xno = aregion(X)
        Yno = aregion(Y)

        # Name variables
        short_channel = channel[:2].upper()
        hist_title = ("Likelihood ratio distribution of no change region {} ENL={}"
            .format(short_channel, ENL))
        hist_filename = "fig/gamma/gamma.hist.ENL{0}.{1}.{2}.pdf".format(ENL, short_channel, percent)

        # No change region histogram
        gno = Gamma(Xno, Yno, ENL, ENL)
        f, ax = gno.histogram(percent)
        # ax.set_title(hist_title)
        f.savefig(hist_filename, bbox_inches='tight')

        ## Images
        g = Gamma(X, Y, ENL, ENL)

        # Binary image
        im = g.image_binary(percent)
        im_filename = "fig/gamma/gamma.im.ENL{0}.{1}.{2}.jpg".format(ENL, short_channel, percent)
        plt.imsave(im_filename, im, cmap='gray')

        # Two level image, grayscale
        im = g.image_color2(percent)
        im_filename = "fig/gamma/gamma.color2.ENL{0}.{1}.{2}.jpg".format(ENL, short_channel, percent)
        plt.imsave(im_filename, im, cmap="gray", vmin=0, vmax=1)

        # Two level image, color
        im = g.image_color3(percent)
        im_filename = "fig/gamma/gamma.color3.ENL{0}.{1}.{2}.jpg".format(ENL, short_channel, percent)
        plt.imsave(im_filename, im)

    gamma_test(april, may, "hhhh", 13, 0.10)
    gamma_test(april, may, "hvhv", 13, 0.10)
    gamma_test(april, may, "vvvv", 13, 0.10)

    gamma_test(april, may, "hhhh", 13, 0.05)
    gamma_test(april, may, "hvhv", 13, 0.05)
    gamma_test(april, may, "vvvv", 13, 0.05)

    gamma_test(april, may, "hhhh", 13, 0.01)
    gamma_test(april, may, "hvhv", 13, 0.01)
    gamma_test(april, may, "vvvv", 13, 0.01)

    gamma_test(april, may, "hhhh", 13, 0.001)
    gamma_test(april, may, "hvhv", 13, 0.001)
    gamma_test(april, may, "vvvv", 13, 0.001)

    gamma_test(april, may, "hhhh", 13, 0.0001)
    gamma_test(april, may, "hvhv", 13, 0.0001)
    gamma_test(april, may, "vvvv", 13, 0.0001)

    gamma_test(april, may, "hhhh", 13, 0.00001)
    gamma_test(april, may, "hvhv", 13, 0.00001)
    gamma_test(april, may, "vvvv", 13, 0.00001)

    # At lower ENL than normal
    gamma_test(april, may, "hhhh", 12, 0.01)
    gamma_test(april, may, "hvhv", 12, 0.01)
    gamma_test(april, may, "vvvv", 12, 0.01)

    f, ax = multiENL_gamma(april_no_change.hhhh, may_no_change.hhhh)
    f.savefig("fig/gamma/gamma.multiENL.HH.pdf", bbox_inches='tight')

    f, ax = multiENL_gamma(april_no_change.hvhv, may_no_change.hvhv)
    f.savefig("fig/gamma/gamma.multiENL.HV.pdf", bbox_inches='tight')

    f, ax = multiENL_gamma(april_no_change.vvvv, may_no_change.vvvv)
    f.savefig("fig/gamma/gamma.multiENL.VV.pdf", bbox_inches='tight')

    f, ax = critical_region()
    f.savefig("fig/gamma/gamma.critical-region.pdf", bbox_inches='tight')

