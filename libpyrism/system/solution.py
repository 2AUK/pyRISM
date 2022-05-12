#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, field

@dataclass
class RISMSolution:
    direct: np.ndarray
    indirect: np.ndarray
    total: np.ndarray
