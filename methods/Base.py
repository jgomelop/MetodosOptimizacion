from abc import abstractmethod
from abc import ABCMeta

class IBaseMethod(metaclass=ABCMeta):
    @abstractmethod
    def plot_function(self):
        pass

    @abstractmethod
    def plot_error(self):
        pass

    @abstractmethod
    def get_relative_error(self):
        pass

class IRootFindingMethod(metaclass=ABCMeta):
    @abstractmethod
    def get_estimated_root(self):
        pass

class IOptimizationMethod(metaclass=ABCMeta):
    @abstractmethod
    def get_estimated_opt(self):
        pass