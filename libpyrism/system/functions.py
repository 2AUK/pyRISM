#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, field

@dataclass
class RISMFunctions:
    """
    Class to collate all the functions used in a RISMProblem.
    Also keeps track of the sizes of each array
    """
    size: tuple
    omega: np.ndarray
    pot: np.ndarray = field(init=False)
    pot_short: np.ndarray = field(init=False)
    pot_long: np.ndarray = field(init=False)
    direct: np.ndarray = field(init=False)
    indirect: np.ndarray = field(init=False)
    total: np.ndarray = field(init=False)

    def __post_init__(self):
        pot = np.zeros(size, dtype=np.float64)
        pot_short = np.zeros(size, dtype=np.float64)
        pot_long = np.zeros(size, dtype=np.float64)
        direct = np.zeros(size, dtype=np.float64)
        indirect = np.zeros(size, dtype=np.float64)
        total = np.zeros(size, dtype=np.float64)
