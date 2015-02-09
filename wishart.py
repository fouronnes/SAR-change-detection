import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import determinant, sar_sum

def block_diagonal(X, Y, n, m):
    p = 3
    detX = X.hhhh*X.hvhv*X.vvvv
    detY = Y.hhhh*Y.hvhv*Y.vvvv
    detXY = (X.hhhh+Y.hhhh)*(X.hvhv+Y.hvhv)*(X.vvvv+Y.vvvv)

    lnq = (p*(n+m)*np.log(n+m) - p*n*np.log(n) - p*m*np.log(m)
            + n*np.log(detX) + m*np.log(detY) - (n+m)*np.log(detXY))
    # same as full covariance ??
    rho = 1 - (2*p*p - 1)/(6*p) * (1/n + 1/m - 1/(n+m))
    w2 = (-(p*p/4)*(1-1/rho)**2 + p*p*(p*p - 1)/24 * (1/(n*n) + 1/(m*m) - 1/((n+m)**2))*1/(p*p))

    return lnq, rho, w2

def azimuthal_symmetry(X, Y, n, m):
    p1 = 2
    p2 = 1
    p = np.sqrt(p1**2 + p2**2)

    detX = np.real(X.hvhv*(X.hhhh*X.vvvv - X.hhvv*np.conj(X.hhvv)))
    detY = np.real(Y.hvhv*(Y.hhhh*Y.vvvv - Y.hhvv*np.conj(Y.hhvv)))
    detXY = np.real((X.hvhv+Y.hvhv) * ((X.hhhh+Y.hhhh)*(X.vvvv+Y.vvvv) - (X.hhvv+Y.hhvv)*(np.conj(X.hhvv)+np.conj(Y.hhvv))))

    lnq = (p*(n+m)*np.log(n+m) - p*n*np.log(n) - p*m*np.log(m)
            + n*np.log(detX) + m*np.log(detY) - (n+m)*np.log(detXY))

    rho1 = 1 - (2*p1**2 - 1)/(6*p1) * (1/n + 1/m - 1/(n+m))
    rho2 = 1 - (2*p2**2 - 1)/(6*p2) * (1/n + 1/m - 1/(n+m))
    rho = 1/p**2 * (p1**2 * rho1 + p2**2 * rho2)

    w2 = - p**2/4 * (1-1/rho)**2 + (p1**2*(p1**2-1) + p2**2*(p2**2-1))/24 * (1/n**2 + 1/m**2 - 1/(n+m)**2) * 1/rho**2

    return lnq, rho, w2

def full_covariance(X, Y, n, m):
    p = 3
    detX = determinant(X)
    detY = determinant(Y)
    detXY = determinant(sar_sum([X, Y]))

    lnq = (p*(n+m)*np.log(n+m) - p*n*np.log(n) - p*m*np.log(m)
            + n*np.log(detX) + m*np.log(detY) - (n+m)*np.log(detXY))
    rho = 1 - (2*p*p - 1)/(6*p) * (1/n + 1/m - 1/(n+m))
    w2 = (-(p*p/4)*(1-1/rho)**2 + p*p*(p*p - 1)/24 * (1/(n*n) + 1/(m*m) - 1/((n+m)**2))*1/(rho*rho))

    return lnq, rho, w2

class Wishart(object):
    def __init__(self, X, Y, n, m, mode):
        self.X = X
        self.Y = Y
        self.n = n
        self.m = m
        self.mode = mode

        if mode == "diagonal":
            self.lnq, self.rho, self.w2 = block_diagonal(X, Y, n, m)
        elif mode == "azimuthal":
            self.lnq, self.rho, self.w2 = azimuthal_symmetry(X, Y, n, m)
        elif mode == "full":
            self.lnq, self.rho, self.w2 = full_covariance(X, Y, n, m)
        else:
            raise RuntimeError("Invalid Wishart test mode:" + repr(mode))

    def histogram(self, percent):
        f = plt.figure(figsize=(8, 4))
        ax = f.add_subplot(111)
        ax.hist(-2*self.lnq.flatten(), bins=100, normed=True, color="#3F5D7D")

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
        im[-2*self.lnq > threshold] = 1
        return im

    def image_linear(self, p1, p2):
        # Select thresholds from chi2 percentile (ignore w2 term)
        p = 3
        chi2 = scipy.stats.chi2(p**2)
        t1 = chi2.ppf(1.0 - p1)
        t2 = chi2.ppf(1.0 - p2)

        return matplotlib.colors.normalize(t1, t2, clip=True)(-2*self.lnq)

