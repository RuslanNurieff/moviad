from abc import abstractmethod

class Logger:
    @abstractmethod
    def log(self, message: str):
        pass