import sys
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *
from plotting import *
from gamma import *
from wishart import *

# Ranges defining the forest region of the image
no_change_i = range(307, 455)
no_change_j = range(52, 120)

def aregion(X):
    return X[np.ix_(no_change_i, no_change_j)]

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

def critical_region_wishart():
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

    ax.set_ylim([0, 0.13])

    # Chi2
    ENL = 13
    p = 3
    chi2 = scipy.stats.chi2(p**2)
    x = np.linspace(0, 25, 500)
    y = chi2.pdf(x)
    ax.plot(x, y, color='black', linewidth=1)
    
    # Thresholds
    t = chi2.ppf(1 - percent)
    ax.fill_between(x, y, where=(x < t), color='#3F5D7D')

    ax.set_xticks([t])
    ax.set_xticklabels([r"$T$"], size=16)

    anotx = 7.05
    ax.annotate(r'$\chi^2(p^2)$', xy=(anotx, chi2.pdf(anotx)), xytext=(anotx + 6, chi2.pdf(anotx)),
        arrowprops=dict(facecolor='black', shrink=0.05, width=.3, headwidth=5),
        fontsize=16,
        horizontalalignment='right',
        verticalalignment='center'
        )

    ax.text(4.5, 0.04, "No change", color="white", size=16)
    ax.text(16, 0.04, "Change", color="black", size=16)

    ax.axvline(t, color="black", linestyle="--")

    return f, ax

if __name__ == "__main__":
    # Load data
    april = SARData().load_april()
    may = SARData().load_may()

    # No change region
    april_no_change = region(april, no_change_i, no_change_j)
    may_no_change = region(may, no_change_i, no_change_j)

    # Color composites
    plt.imsave("fig/april.jpg", color_composite(april))
    plt.imsave("fig/may.jpg", color_composite(may))

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
        hist_filename = "fig/gamma/gamma.hist.ENL{0}.{1}.{2}.pdf".format(ENL, short_channel, percent)
        im_filename = "fig/gamma/gamma.im.ENL{0}.{1}.{2}.jpg".format(ENL, short_channel, percent)

        # No change region histogram
        gno = Gamma(Xno, Yno, ENL, ENL)
        f, ax = gno.histogram(percent)
        # ax.set_title(hist_title)
        f.savefig(hist_filename, bbox_inches='tight')

        # Binary image
        g = Gamma(X, Y, ENL, ENL)
        im = g.image_binary(percent)
        plt.imsave(im_filename, im, cmap='gray')

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

    ## Wishart

    def wishart_test(mode, ENL, percent):
        # Test statistic over the whole area
        w = Wishart(april, may, ENL, ENL, mode)

        # Test statistic over the no change region
        wno = Wishart(april_no_change, may_no_change, ENL, ENL, mode)

        # Histogram, no change region
        f, ax = wno.histogram(percent)
        hist_filename = "fig/wishart/{}/lnq.hist.ENL{}.pdf".format(mode, ENL)
        f.savefig(hist_filename, bbox_inches='tight')

        # Histogram, entire region
        f, ax = w.histogram(percent)
        hist_filename = "fig/wishart/{}/lnq.hist.total.ENL{}.pdf".format(mode, ENL)
        f.savefig(hist_filename, bbox_inches='tight')

        # Binary image
        im = w.image_binary(percent)
        plt.imsave("fig/wishart/{}/lnq.ENL{}.{}.jpg".format(mode, ENL, percent), im, cmap="gray")

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
    plt.imsave("fig/wishart/lnq.linear.jpg", im, cmap="gray")

    # Rho and omega2 plots
    f, ax = rho_plot()
    f.savefig("fig/wishart/rho.pdf", bbox_inches='tight')
    f, ax = omega2_plot()
    f.savefig("fig/wishart/omega2.pdf", bbox_inches='tight')

    # Wishart critical region figure
    f, ax = critical_region_wishart()
    f.savefig("fig/wishart/wishart.critical-region.pdf", bbox_inches='tight')

    plt.close('all')

