from abc import ABC, abstractmethod

class baseLLPClassifier(ABC):
    @abstractmethod
    def fit(self, X, bags, proportions):
        pass

    @abstractmethod
    def predict(self, X):
        pass
