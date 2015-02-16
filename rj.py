import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sar_data import *
from omnibus import Omnibus

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
        self.H = [None]*(k-1)

        # Hypothesis K[l, s] is
        # S(l+s) = S(l+s-1)
        self.K = np.zeros((k-1, k-1))
        self.K[:,:] = np.nan

        for l in range(0, k-1):
            self.H[l] = Omnibus(sar_list[l:], ENL)

            for s in range(1, k-l):
                pass

if __name__ == "__main__":

    def print_pvalue_table(rj):
        "Pretty-print the table of p-values"
        print("""
        {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.4f}
        """.format(
            rj.H[0].pvalue(), rj.H[1].pvalue(), rj.H[2].pvalue(), rj.H[3].pvalue(), rj.H[4].pvalue()
        ))

    print("Rj test...")
    # rj_all = RjTest(sar_list, 13)

    print("Forest:")
    rj_nochange = RjTest(sar_list_nochange, 13)
    print_pvalue_table(rj_nochange)

    print("Rye:")
    rj_rye = RjTest(sar_list_rye, 13)
    print_pvalue_table(rj_rye)

    print("Grass:")
    rj_grass = RjTest(sar_list_grass, 13)
    print_pvalue_table(rj_grass)


