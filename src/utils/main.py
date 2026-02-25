from typing import Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum
from collections import Counter
from time import process_time_ns

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, brier_score_loss, log_loss as sk_log_loss
from imblearn.metrics import geometric_mean_score


Dataset = Tuple[np.ndarray, np.ndarray]  # data, labels


def get_predictions(estimator, X, proba: bool = False) -> Tuple[np.ndarray, int]:
    start = process_time_ns()
    predictions = estimator.predict(X) if not proba else estimator.predict_proba(X)
    stop = process_time_ns()

    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = predictions[:, 1]

    predictions = np.array([predictions]).T
    desired_shape = (X.shape[0], 1)
    if predictions.shape != desired_shape:
        predictions = predictions.reshape(desired_shape)

    return predictions.ravel(), stop - start


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def divide_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[Dataset, Dataset]:
    """ Returns ((x_majority, y_majority), (x_minority, y_minority)) """

    positive_samples_indices = np.where(y == 1)[0]
    negative_samples_indices = np.where(y == 0)[0]
    positive_count = len(positive_samples_indices)
    negative_count = len(negative_samples_indices)

    if positive_count > negative_count:
        x_majority = X[positive_samples_indices, :]
        y_majority = y[positive_samples_indices]
        x_minority = X[negative_samples_indices, :]
        y_minority = y[negative_samples_indices]
    else:
        x_majority = X[negative_samples_indices, :]
        y_majority = y[negative_samples_indices]
        x_minority = X[positive_samples_indices, :]
        y_minority = y[positive_samples_indices]

    return (x_majority, y_majority), (x_minority, y_minority)

def set_majority_as_one(y_majority: np.ndarray, y_minority: np.ndarray) -> Dataset:
    y_majority = np.ones(y_majority.shape)
    y_minority = np.zeros(y_minority.shape)

    return y_majority, y_minority

def f1_measure(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)

    return f1_score(y, predictions), time_elapsed

def auc_score(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)

    return roc_auc_score(y, predictions), time_elapsed

def g_mean(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)

    return geometric_mean_score(y, predictions), time_elapsed

def brier_score(estimator, X, y, proba: bool = True) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X, proba)

    return brier_score_loss(y, predictions), time_elapsed

def log_loss(estimator, X, y, proba: bool = True) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X, proba)

    return sk_log_loss(y, predictions), time_elapsed

class VotingType(Enum):
    MEAN = 'mean'
    MAJORITY = 'majority'

class OptimizationType(Enum):
    ROC = 'roc_curve'
    NONE = 'none'

class ThresholdOptimizer(ABC):
    def __init__(self):
        self._threshold: Optional[float] = None

    @abstractmethod
    def optimize(self, y: np.ndarray, scores: np.ndarray) -> float:
        raise NotImplemented

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("You must first optimize threshold!")

        return self._threshold

class ROCOptimizer(ThresholdOptimizer):
    """
    Using Youden's J statistic
    """

    def optimize(self, y: np.ndarray, scores: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(y, scores)
        J = tpr - fpr
        self._threshold = thresholds[np.argmax(J)]

        return self._threshold

class FixedOptimizer(ThresholdOptimizer):
    def __init__(self, threshold: float = .5):
        super().__init__()
        self._threshold = threshold

    def optimize(self, y: np.ndarray, scores: np.ndarray) -> float:
        return self._threshold

class Voting(ABC):
    @abstractmethod
    def vote(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        raise NotImplemented()

class MeanVoting(Voting):
    def vote(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        mean_scores = scores.mean(axis=0)

        return np.where(mean_scores >= threshold, 1, 0)

class MajorityVoting(Voting):
    def vote(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        predictions = []
        for observation_i in range(scores.shape[1]):
            counter = Counter(scores[:, observation_i])
            most_common = counter.most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

class MajorityVotingFromProba(Voting):
    def vote(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        classifications = np.array([np.where(scores[cls, :] >= threshold, 1, 0) for cls in range(scores.shape[0])])
        majority_voting = MajorityVoting()

        return majority_voting.vote(classifications, threshold)

class VotingFactory:
    @staticmethod
    def get(voting: VotingType, predict_proba: bool = False) -> Voting:
        if predict_proba and voting == VotingType.MAJORITY:
            return MajorityVotingFromProba()

        return {
            VotingType.MEAN: MeanVoting(),
            VotingType.MAJORITY: MajorityVoting()
        }[voting]

class ThresholdOptimizerFactory:
    @staticmethod
    def get(optimization: OptimizationType) -> ThresholdOptimizer:
        return {
            OptimizationType.ROC: ROCOptimizer(),
            OptimizationType.NONE: FixedOptimizer()
        }[optimization]

def choices(sequence: np.ndarray, k: int) -> np.ndarray:
    indices = np.random.choice(len(sequence), size=k, replace=True)

    return sequence[indices]
