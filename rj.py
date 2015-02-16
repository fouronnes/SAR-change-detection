import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *

def omnibus_lnq(sar_list, n):
    p = 3
    k = len(sar_list)
    f = (k-1)*p**2

    sum_term = sum([np.log(determinant(Xi)) for Xi in sar_list])
    X = sar_sum(sar_list)
    return n*(p*k*np.log(k) + sum_term - k*np.log(determinant(X)))

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
        n = ENL

        # Hypothesis H[l] is
        # Sl = ... = Sk
        self.H = 0

        # Hypothesis K[l, s] is
        # S(l+s) = S(l+s-1)
        self.K = 0

        for l in range(0, k-1):
            lnq = omnibus_lnq(sar_list[l:], ENL)

            for s in range(1, k-l):
                pass

if __name__ == "__main__":
    print("Rj test...")
    rj = RjTest(sar_list, 13)
