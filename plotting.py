import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors
from sar_data import *

def sar_show(channel):
    "Grayscale image of a single SAR channel"
    f, ax = plt.subplots(1, 1)
    ax.imshow(np.log(channel), cmap="gray", vmin=-10, vmax=0)

def color_composite(X):
    "Color composite of a EMISAR image"

    green = 10*np.log(X.hhhh) / np.log(10)
    blue = 10*np.log(X.vvvv) / np.log(10)
    red = 10*np.log(X.hvhv) / np.log(10)

    # Normalize
    green = matplotlib.colors.normalize(-30, 0, clip=True)(green)
    blue = matplotlib.colors.normalize(-30, 0, clip=True)(blue)
    red = matplotlib.colors.normalize(-36, -6, clip=True)(red)

    return np.concatenate((red[:,:,None], green[:,:,None], blue[:,:,None]), axis=2)

