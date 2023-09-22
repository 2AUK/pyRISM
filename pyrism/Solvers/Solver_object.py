from pyrism.Core import RISM_Obj
import numpy as np
from dataclasses import dataclass, field
from numba import njit


@dataclass
class SolverObject:
    data_vv: RISM_Obj
    tol: float
    max_iter: int
    damp_picard: float
    rms: float = field(default=0.0)
    data_uv: RISM_Obj = field(default=None)

    def step_Picard(self, curr, prev):
        return prev + self.damp_picard * (curr - prev)

    def converged(self, curr, prev):
        self.rms = converged_impl(
            curr, prev, self.data_vv.grid.d_r, np.prod(curr.shape)
        )

        if self.rms < self.tol:
            return True
        else:
            return False

    def epilogue(self, curr_iter, nlam):
        print(
            """Current Lambda: {nlam}\nTotal Iterations: {curr_iter}\nRMS: {rms}""".format(
                nlam=nlam, curr_iter=curr_iter, rms=self.rms
            )
        )


@njit
def converged_impl(curr, prev, dr, denom):
    return np.sqrt(dr * np.power((curr - prev), 2).sum() / denom)
