import numpy as np
from .XRISM import XRISM


class IntegralEquation(object):
    IE_dispatcher = {"XRISM": XRISM}

    def __init__(self, IE):
        self.IE = IE

    def get_IE(self):
        return self.IE_dispatcher[self.IE]
