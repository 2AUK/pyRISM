#!/usr/bin/env python3

from abc import abstractmethod, ABC
from grid import Grid
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Potential(ABC):
    grid: Grid
    params: list = field(init=False, default_factory=list)
    func: np.ndarray = field(init=False)

    @abstractmethod
    def _calculate(self):
        raise NotImplementedError

    @abstractmethod
    def _mixing(self):
        raise NotImplementedError

    @abstractmethod
    def add_param(self):
        raise NotImplementedError

    @abstractmethod
    def tabulate(self):
        raise NotImplementedError
