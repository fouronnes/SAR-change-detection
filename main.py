import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *
from plotting import *

# Ranges defining the forest region of the image
no_change_i = range(307, 455)
no_change_j = range(52, 120)

def aregion(X):
    return X[np.ix_(no_change_i, no_change_j)]

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

def block_diagonal(X, Y, n, m):
    p = 3
    detX = X.hhhh*X.hvhv*X.vvvv
    detY = Y.hhhh*Y.hvhv*Y.vvvv
    detXY = (X.hhhh+Y.hhhh)*(X.hvhv+Y.hvhv)*(X.vvvv+Y.vvvv)
    return (p*(n+m)*np.log(n+m) - p*n*np.log(n) - p*m*np.log(m)
            + n*np.log(detX) + m*np.log(detY) - (n+m)*np.log(detXY))

def azimuthal_symmetry(X, Y, n, m):
    p = 3
    detX = np.real(X.hvhv*(X.hhhh*X.vvvv - X.hhvv*np.conj(X.hhvv)))
    detY = np.real(Y.hvhv*(Y.hhhh*Y.vvvv - Y.hhvv*np.conj(Y.hhvv)))
    detXY = np.real((X.hvhv+Y.hvhv) * ((X.hhhh+Y.hhhh)*(X.vvvv+Y.vvvv) - (X.hhvv+Y.hhvv)*(np.conj(X.hhvv)+np.conj(Y.hhvv))))
    return (p*(n+m)*np.log(n+m) - p*n*np.log(n) - p*m*np.log(m)
            + n*np.log(detX) + m*np.log(detY) - (n+m)*np.log(detXY))

def full_covariance(X, Y, n, m):
    p = 3
    detX = determinant(X)
    detY = determinant(Y)
    detXY = determinant(sar_sum(X, Y))
    return (p*(n+m)*np.log(n+m) - p*n*np.log(n) - p*m*np.log(m)
            + n*np.log(detX) + m*np.log(detY) - (n+m)*np.log(detXY))

class Wishart(object):
    def __init__(self, X, Y, n, m, mode):
        self.X = X
        self.Y = Y
        self.n = n
        self.m = m
        self.mode = mode

        if mode == "diagonal":
            self.lnq = block_diagonal(X, Y, n, m)
        elif mode == "azimuthal":
            self.lnq = azimuthal_symmetry(X, Y, n, m)
        elif mode == "full":
            self.lnq = full_covariance(X, Y, n, m)
        else:
            raise RuntimeError("Invalid Wishard test mode:" + repr(mode))

        p = 3
        self.rho = 1 - (2*p*p - 1)/(6*p) * (1/n + 1/m - 1/(n+m))

        self.w2 = (-(p*p/4)*(1-1/self.rho)**2
                + p*p*(p*p - 1)/24 * (1/(n*n) + 1/(m*m) - 1/((n+m)**2))*1/(p*p))

    def histogram(self, percent):
        f = plt.figure(figsize=(8, 4))
        ax = f.add_subplot(111)
        ax.hist(-2*self.rho*self.lnq.flatten(), bins=100, normed=True, color="#3F5D7D")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Overlay pdf
        p = 3
        x = np.linspace(0, 40, 1000)
        chi2 = scipy.stats.chi2
        y = chi2.pdf(x, p**2) + self.w2*(chi2.pdf(x, p**2+4) - chi2.pdf(x, p**2))
        ax.plot(x, y, color="black", linewidth=2)

        ax.set_xlim([0, 40])

        return f, ax

    def image_binary(self, percent):
        # Select threshold from chi2 percentile (ignore w2 term)
        p = 3
        chi2 = scipy.stats.chi2(p**2)
        threshold = chi2.ppf(1.0 - percent)

        im = np.zeros_like(self.lnq)
        im[-2*self.rho*self.lnq > threshold] = 1
        return im

    def image_linear(self, p1, p2):
        # Select threshold from chi2 percentile (ignore w2 term)
        p = 3
        chi2 = scipy.stats.chi2(p**2)
        t1 = chi2.ppf(1.0 - p1)
        t2 = chi2.ppf(1.0 - p2)

        return matplotlib.colors.normalize(t1, t2, clip=True)(-2*self.rho*self.lnq)

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
    def overlay(ENL):
        F = scipy.stats.f(2*ENL, 2*ENL)
        x = np.linspace(0, 3, 500)
        ax.plot(x, F.pdf(x), color='black', linewidth=1)
        mode = (ENL - 1)/(ENL+1)
        xtext = 0.4 if ENL%2 == 0 else 1.3
        ax.annotate('{}'.format(ENL), xy=(mode, F.pdf(mode)), xytext=(xtext, F.pdf(mode)),
            arrowprops=dict(facecolor='black', shrink=0.05, width=.5, headwidth=2),
            fontsize=11,
            horizontalalignment='right',
            verticalalignment='center'
            )

    overlay(8)
    overlay(9)
    overlay(10)
    overlay(11)
    overlay(12)
    overlay(13)
    overlay(14)
    overlay(15)

    return f, ax

if __name__ == "__main__":

    # Load data
    april = SARData().load_april()
    may = SARData().load_may()

    # No change region
    april_no_change = region(april, no_change_i, no_change_j)
    may_no_change = region(may, no_change_i, no_change_j)

    # Color composites
    plt.imsave("fig/april.png", color_composite(april))
    plt.imsave("fig/may.png", color_composite(may))

    ## Gamma

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
        hist_filename = "fig/gamma/gamma.hist.ENL{0}.{1}.{2:.2f}.pdf".format(ENL, short_channel, percent)
        im_filename = "fig/gamma/gamma.im.ENL{0}.{1}.{2:.2f}.png".format(ENL, short_channel, percent)

        # No change region histogram
        gno = Gamma(Xno, Yno, ENL, ENL)
        f, ax = gno.histogram(percent)
        # ax.set_title(hist_title)
        f.savefig(hist_filename, bbox_inches='tight')

        # Binary image
        g = Gamma(X, Y, ENL, ENL)
        im = g.image_binary(percent)
        plt.imsave(im_filename, im, cmap='gray')

    gamma_test(april, may, "hhhh", 13, 0.01)
    gamma_test(april, may, "hvhv", 13, 0.01)
    gamma_test(april, may, "vvvv", 13, 0.01)

    gamma_test(april, may, "hhhh", 12, 0.01)
    gamma_test(april, may, "hvhv", 12, 0.01)
    gamma_test(april, may, "vvvv", 12, 0.01)

    f, ax = multiENL_gamma(april_no_change.hhhh, may_no_change.hhhh)
    f.savefig("fig/gamma/gamma.multiENL.HH.pdf", bbox_inches='tight')

    ## Wishart

    def wishart_test(mode, ENL, percent):
        w = Wishart(april, may, ENL, ENL, mode)
        wno = Wishart(april_no_change, may_no_change, ENL, ENL, mode)

        f, ax = wno.histogram(percent)
        hist_title = (r"$-2 \rho \ln Q$ distribution in no change region ENL={}"
                .format(ENL))
        hist_filename = "fig/wishart/{}/lnq.hist.ENL{}.pdf".format(mode, ENL)

        # ax.set_title(hist_title)
        f.savefig(hist_filename, bbox_inches='tight')

        im = w.image_binary(percent)
        plt.imsave("fig/wishart/{}/lnq.ENL{}.{}.png".format(mode, ENL, percent), im, cmap="gray")

    wishart_test("full", 13, 0.00001)
    wishart_test("full", 13, 0.0001)
    wishart_test("full", 13, 0.001)
    wishart_test("full", 13, 0.01)
    wishart_test("full", 13, 0.05)
    wishart_test("full", 13, 0.10)

    wishart_test("full", 11, 0.01)
    wishart_test("full", 12, 0.01)
    wishart_test("full", 13, 0.01)
    wishart_test("full", 14, 0.01)

    wishart_test("diagonal", 13, 0.00001)
    wishart_test("diagonal", 13, 0.0001)
    wishart_test("diagonal", 13, 0.001)
    wishart_test("diagonal", 13, 0.01)
    wishart_test("diagonal", 13, 0.05)
    wishart_test("diagonal", 13, 0.10)

    wishart_test("azimuthal", 13, 0.00001)
    wishart_test("azimuthal", 13, 0.0001)
    wishart_test("azimuthal", 13, 0.001)
    wishart_test("azimuthal", 13, 0.01)
    wishart_test("azimuthal", 13, 0.05)
    wishart_test("azimuthal", 13, 0.10)

    w = Wishart(april, may, 13, 13, "full")
    im = w.image_linear(0.01, 0.00001)
    plt.imsave("fig/wishart/lnq.linear.png", im, cmap="gray")

    plt.close('all')

