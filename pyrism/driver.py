import numpy as np
import grid


class driver(object):
    def __init__(self, RISM_Obj):
        self.grid = grid.Grid(RISM_Obj.npts, RISM_Obj.radius)

    def RISM(self, N, M, wk, cr, vklr, rho):
        """
        Computes RISM equation in the form:

        h = w*c*(wcp)^-1*w*c

        Parameters
        ----------
        wk: Intramolecular correlation function 3D-array

        cr: Direct correlation function 3D-array

        vrlr: Long-range potential

        rho: Number density matrix

        Returns
        -------
        trsr: ndarray
        An array containing short-range indirection correlation function
        """
        I = np.eye(N, M=M)
        h = np.zeros((self.grid.npts, N, M), dtype=np.float64)
        trsr = np.zeros((self.grid.npts, N, M), dtype=np.float64)
        cksr = np.zeros((self.grid.npts, N, M), dtype=np.float64)

        for i, j in np.ndindex(N, M):
            cksr[:, i, j] = self.grid.dht(cr[:, i, j])
            cksr[:, i, j] -= vklr[:, i, j]
        for i in range(self.grid.npts):
            iwcp = np.linalg.inv(I - wk[i, :, :] @ cksr[i, :, :] @ rho)
            wcw = wk[i, :, :] @ cksr[i, :, :] @ wk[i, :, :]
            h[i, :, :] = iwcp @ wcw - cksr[i, :, :]
        for i, j in np.ndindex(N, M):
            trsr[:, i, j] = self.grid.idht(h[:, i, j] - vklr[:, i, j])
        return trsr