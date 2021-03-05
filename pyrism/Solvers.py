import numpy as np
import mpmath as mp
from scipy.fft import dst, idst
from scipy.special import erf, expit
import matplotlib.pyplot as plt
import grid

def picard_step(cr_cur, cr_prev, damp):
    return damp*cr_cur + (1-damp)*cr_prev

def ng_step(n_pic=2, cr_cur, cr_prev, damp):
    pass
