from Core import RISM_Obj
import numpy as np


class SolverObject(object):
    def __init__(self, data, tol, max_iter, damp_picard=0.01):
        self.data = data
        self.tol = tol
        self.max_iter = max_iter
        self.damp_picard = damp_picard
        self.rms = None

    def step_Picard(self, curr, prev):
        return prev + self.damp_picard * (curr - prev)

    def converged(self, curr, prev):
        self.rms = np.sqrt(
            data.grid.d_r * np.power((curr - prev), 2).sum() / (np.prod(curr.shape))
        )

        if self.rms < self.tol:
            return True
        else:
            return False

    def epilogue(self, curr_iter, nlam):
        print(
            """Current Lambda: {nlam}
            Total Iterations: {curr_iter}
            RMS: {rms}""".format(
                nlam=nlam, curr_iter=curr_iter, rms=self.rms
            )
        )
