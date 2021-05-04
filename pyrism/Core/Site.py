import numpy as np
import attr


@attr.s
class Site(object):

    atom_type: str = attr.ib()
    params: list = attr.ib()  # Varies depending on Potential type
    p: float = attr.ib()
