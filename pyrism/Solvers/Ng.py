import numpy as np
import Data
import attr
from Solver import *

@attr.s
class NgSolver(SolverObject):

    fr: list = attr.ib(default=[])
    gr: list = attr.ib(default=[])

    def step_Picard(self, curr, prev):
        self.fr.append(prev)
        self.gr.append(curr)
        return prev + self.damp_picard * (curr - prev)

    def step_Ng(self, curr, prev):
        vecdr = np.asarray(self.gr) - np.asarray(self.fr)
        dn = vecdr[-1].flatten()
        d01 = (vecdr[-1] - vecdr[-2]).flatten()
        d02 = (vecdr[-1] - vecdr[-3]).flatten()
        A[0, 0] = np.inner(d01, d01)
        A[0, 1] = np.inner(d01, d02)
        A[1, 0] = np.inner(d01, d02)
        A[1, 1] = np.inner(d02, d02)
        b[0] = np.inner(dn, d01)
        b[1] = np.inner(dn, d02)
        c = np.linalg.solve(A, b)
        c_next = (
            (1 - c[0] - c[1]) * self.gr[-
            + c[0] * self.gr[-2]
            + c[1] * self.gr[-3]
        )
        self.fr.append(prev)
        self.gr.append(curr)
        self.gr.pop(0)
        self.fr.pop(0)
        return c_next

    def solve(self, RISM, Closure):
        i: int = 0
        A = np.zeros((2, 2), dtype=np.float64)
        b = np.zeros(2, dtype=np.float64)

        print("\nSolving RISM equation...\n")

        while i < self.max_iter:
            c_prev = data.c
            RISM(self.data)
            c_A = Closure(self.data)

            if i < 3:
                c_next = self.step_Picard(c_A, c_prev)
            else:
                c_next = self.step_Ng(c_A, c_prev)

            if self.converged:
                epilogue(i, data.nlam)
                break

            i += 1

            if i == self.max_iter:
                print("Max iteration reached!")
                epilogue(i, data.nlam)
                break

            data.c = c_next
        
        data.c -= (self.B * data.uk_lr)
        data.t += (self.B * data.uk_lr)