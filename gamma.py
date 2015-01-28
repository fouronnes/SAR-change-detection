import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors

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

        # ax.axvline(t_inf, 0, 1, color='black', linestyle='--')
        # ax.text(t_inf, 1.12, " 5% percentile")
        # ax.axvline(t_sup, 0, 1, color='black', linestyle='--')
        # ax.text(t_sup, 1.12, " 95% percentile")

        return f, ax
    
    def image_binary(self, percentile):
        F = scipy.stats.f(2*self.m, 2*self.n)
        t_inf, t_sup = F.ppf(percentile/2), F.ppf(1 - percentile/2)

        im = np.zeros_like(self.Q)
        im[self.Q < t_inf] = 1
        im[self.Q > t_sup] = 1

        return im

    def image_linear(self, percentile):
        pass

