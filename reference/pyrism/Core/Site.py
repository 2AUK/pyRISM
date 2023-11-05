import numpy as np
from dataclasses import dataclass, field


@dataclass
class Site(object):
    atom_type: str
    params: list
    coords: np.ndarray
