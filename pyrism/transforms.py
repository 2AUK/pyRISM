#!/usr/bin/env python3
"""
transforms.py

Implementation of the forward and backward fourier-bessel
transforms (Hankel transform) using the discrete sine transform function via scipy.

"""

import numpy as np
from scipy.fftpack import dst, idst
#from mpi4py_fft.fftw import dstn, idstn, FFTW_ESTIMATE


def discrete_hankel_transform(r: 'ndarray', k: 'ndarray', fr: 'ndarray', d_r: float) -> 'ndarray':
    """
    Discrete Hankel Transform

    Parameters
    ----------

    r: ndarray
     Grid to be used over r-space

    k: ndarray
     Grid to be used over k-space

    fr: ndarray
     Function to be transformed from r-space to k-space

    d_r: float
     Grid spacing of r-space

    returns
    -------

    fk: ndarray
     Transformed function from r-space to k-space
    """
    constant =  2 * np.pi * d_r / k
    return constant * dst(fr * r, type=4)

def inverse_discrete_hankel_transform(r: 'ndarray', k: 'ndarray', fk: 'ndarray', d_k: float) -> 'ndarray':
    """
    Inverse Discrete Hankel Transform

    Parameters
    ----------

    r: ndarray
     Grid to be used over r-space

    k: ndarray
     Grid to be used over k-space

    fk: ndarray
     Function to be transformed from k-space to r-space

    d_k: float
     Grid spacing of k-space

    returns
    -------

    fr: ndarray
     Transformed function from k-space to r-space
    """
    constant = d_k / (4*np.pi*np.pi) / r
    return constant * idst(fk * k, type=4)
