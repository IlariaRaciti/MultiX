import matplotlib
matplotlib.use("Agg")
import multiprocessing as mp
import MultiX

############ CODE  ###########
if __name__ == '__main__':
    mp.freeze_support()

    result = MultiX.MultiX("BLCA_normal_CancerBrowser.txt", 20, 0.9, 0.9, 5, "MNDL", 1)