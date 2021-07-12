from abc import ABC, abstractmethod

class Extractor(ABC):
    def __init__(self, **kwargs):
        self.options = {
            'use_integrator' : 0
        }
        self.options.update(kwargs)

    @abstractmethod
    def data(self, experiment):
        raise NotImplementedError()

    @abstractmethod
    def plot(self, experiment, ax, **kwargs):
        raise NotImplementedError()
    
