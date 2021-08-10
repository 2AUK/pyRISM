from Core import RISM_Obj
import numpy as np
from dataclasses import dataclass, field

@dataclass
class SolverObject:
    
    data: RISM_Obj
    tol: float
    max_iter: int
    damp_picard: float
    rms: float = field(default=0.0)

    def step_Picard(self, curr, prev):
        return prev + self.damp_picard * (curr - prev)

    def converged(self, curr, prev):
        self.rms = np.sqrt(
            self.data.grid.d_r * np.power((curr - prev), 2).sum() / (np.prod(curr.shape))
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
