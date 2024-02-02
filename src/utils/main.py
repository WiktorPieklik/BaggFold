from typing import Tuple, Optional
from abc import ABC, abstractmethod
from random import choice
from enum import Enum
from collections import Counter
from time import process_time_ns

import numpy as np
from hmeasure import h_score
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from imblearn.metrics import geometric_mean_score


Dataset = Tuple[np.array, np.array]  # data, labels


def get_predictions(estimator, X) -> Tuple[np.array, int]:
    start = process_time_ns()
    predictions = estimator.predict(X)
    stop = process_time_ns()

    predictions = np.array([predictions]).T
    desired_shape = (X.shape[0], 1)
    if predictions.shape != desired_shape:
        predictions = predictions.reshape(desired_shape)

    return predictions.ravel(), stop - start


def distance(p1: np.array, p2: np.array) -> float:
    return np.linalg.norm(p1 - p2)


def divide_dataset(X: np.array, y: np.array) -> Tuple[Dataset, Dataset]:
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


def h_measure(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)
    n1, n0 = y.sum(), y.shape[0] - y.sum()

    return h_score(y, predictions, severity_ratio=(n1 / n0)), time_elapsed


def f1_measure(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)

    return f1_score(y, predictions), time_elapsed


def auc_score(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)

    return roc_auc_score(y, predictions), time_elapsed


def g_mean(estimator, X, y) -> Tuple[float, int]:
    predictions, time_elapsed = get_predictions(estimator, X)

    return geometric_mean_score(y, predictions), time_elapsed


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
    def optimize(self, y: np.array, scores: np.array) -> float:
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

    def optimize(self, y: np.array, scores: np.array) -> float:
        fpr, tpr, thresholds = roc_curve(y, scores)
        J = tpr - fpr
        self._threshold = thresholds[np.argmax(J)]

        return self._threshold


class FixedOptimizer(ThresholdOptimizer):
    def __init__(self, threshold: float = .5):
        super().__init__()
        self._threshold = threshold

    def optimize(self, y: np.array, scores: np.array) -> float:
        return self._threshold


class Voting(ABC):
    @abstractmethod
    def vote(self, scores: np.array, threshold: float) -> np.array:
        raise NotImplemented()


class MeanVoting(Voting):
    def vote(self, scores: np.array, threshold: float) -> np.array:
        mean_scores = scores.mean(axis=0)

        return np.where(mean_scores >= threshold, 1, 0)


class MajorityVoting(Voting):
    def vote(self, scores: np.array, threshold: float) -> np.array:
        predictions = []
        for observation_i in range(scores.shape[1]):
            counter = Counter(scores[:, observation_i])
            most_common = counter.most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)


class MajorityVotingFromProba(Voting):
    def vote(self, scores: np.array, threshold: float) -> np.array:
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


def choices(sequence: np.array, k: int) -> np.array:
    new_array = np.array([])
    for _ in range(k):
        new_array = np.concatenate((new_array, choice(sequence)))

    return np.reshape(new_array, (-1, sequence.shape[1]))
