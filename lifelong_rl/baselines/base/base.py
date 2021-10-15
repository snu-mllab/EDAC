import abc


class Baseline(object, metaclass=abc.ABCMeta):
    
    """
    General baseline interface.
    """

    @abc.abstractmethod
    def predict(self, path):
        pass
