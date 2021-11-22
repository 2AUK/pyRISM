from .functional_routines import *

class Functional(object):

    functional_dispatcher = {
        "GF": Gaussian_Fluctuations,
        "HNC": HyperNetted_Chain
    }

    def __init__(self, SFE):
        self.SFE = SFE

    def get_functional(self):
        return self.functional_dispatcher[self.SFE]
