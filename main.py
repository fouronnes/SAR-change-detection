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

    # Rho and omega2 plots
    f, ax = rho_plot()
    f.savefig("fig/wishart/rho.pdf", bbox_inches='tight')
    f, ax = omega2_plot()
    f.savefig("fig/wishart/omega2.pdf", bbox_inches='tight')

    plt.close('all')

