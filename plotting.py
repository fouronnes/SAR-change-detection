import numpy as np
import matplotlib.pyplot as plt

def sar_show(channel):
    "Grayscale image of a single SAR channel"
    f, ax = plt.subplots(1, 1)
    ax.imshow(np.log(channel), cmap="gray", vmin=-10, vmax=0)

def rho_plot():
    "Plot of rho in the Wishart test statistic, full covariance case"
    f = plt.figure(figsize=(4, 2))
    ax = f.add_subplot(111)
    
    p = 3
    n = np.linspace(3, 50, 500)
    m = n
    rho = 1 - (2*p*p - 1)/(6*p) * (1/n + 1/m - 1/(n+m))

    ax.plot(n, rho, linewidth=1, color="black")
    ax.set_xlim([0, 50])
    ax.set_xlabel('Number of looks')
    ax.set_ylabel(r'$\rho$', size=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    return f, ax

def omega2_plot():
    "Plot of omega2 in the Wishart test statistic, full covariance case"
    f = plt.figure(figsize=(4, 2))
    ax = f.add_subplot(111)
    
    p = 3
    n = np.linspace(3, 50, 500)
    m = n
    rho = 1 - (2*p*p - 1)/(6*p) * (1/n + 1/m - 1/(n+m))
    w2 = (-(p*p/4)*(1-1/rho)**2 + p*p*(p*p - 1)/24 * (1/(n*n) + 1/(m*m) - 1/((n+m)**2))*1/(rho*rho))

    ax.plot(n, w2, linewidth=1, color="black")
    ax.set_xlim([0, 50])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel('Number of looks')
    ax.set_ylabel(r'$\omega_2$', size=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    return f, ax
