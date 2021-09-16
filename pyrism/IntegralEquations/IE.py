import numpy as np
from .XRISM import XRISM
from .XRISM_UV import XRISM_UV


class IntegralEquation(object):
    IE_dispatcher = {"XRISM": XRISM,
                     "XRISM_UV": XRISM}

    def __init__(self, IE):
        self.IE = IE

    def get_IE(self):
        return self.IE_dispatcher[self.IE]
