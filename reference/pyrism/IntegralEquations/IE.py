import numpy as np
from .XRISM import XRISM
from .DRISM import DRISM
from .XRISM_DB import XRISM_DB


class IntegralEquation(object):
    IE_dispatcher = {
        "XRISM": XRISM,
        "DRISM": DRISM,
        "XRISM-DB": XRISM_DB,
    }

    def __init__(self, IE):
        self.IE = IE

    def get_IE(self):
        return self.IE_dispatcher[self.IE]
