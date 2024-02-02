from abc import ABC, abstractmethod
from math import ceil, floor
from threading import Thread
from typing import Callable, List, Optional, Tuple
from collections import namedtuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import (
    divide_dataset,
    choices,
    VotingType,
    VotingFactory,
    OptimizationType,
    ThresholdOptimizerFactory)

OptimizedCls = namedtuple("OptimizedCls", ["classifier", "optimizer"])


def predict(
        classifiers: List[OptimizedCls],
        X: np.array,
        voting_type: Optional[VotingType] = None,
        predict_proba: bool = False
) -> np.array:
    classifications = np.array([])
    thresholds = []
    for classifier, optimizer in classifiers:
        classifications = np.concatenate(
            (
                classifications,
                classifier.predict(X) if not predict_proba else classifier.predict_proba(X)[:, 1]
            )
        )
        thresholds.append(optimizer.threshold)
    classifications = np.reshape(classifications, (len(classifiers), -1))

    if voting_type:
        voting = VotingFactory.get(voting=voting_type, predict_proba=predict_proba)
        mean_threshold = np.mean(thresholds)

        return voting.vote(classifications, mean_threshold)
    else:
        return classifications


class BaseBaggFold(ABC):
    def __init__(
            self,
            base_classifier_fn: Callable[[bool], object],
            sampler,
            optimization_type: OptimizationType = OptimizationType.ROC,
            voting_type: VotingType = VotingType.MEAN,
            optimization_ds_size: float = .1,
            predict_proba: bool = False,
            random_state: int = 100
    ):
        self._x_majority = None
        self._y_majority = None
        self._x_minority = None
        self._y_minority = None
        self._needed_classificators_count = 0
        self._classifiers: List[OptimizedCls] = []
        self.__init_classifier = base_classifier_fn
        self._sampler = sampler
        self._optimization = optimization_type
        self._voting_type = voting_type
        self._opt_size = optimization_ds_size
        self._predict_proba = predict_proba
        self._random_state = random_state

    @property
    def sampler(self):
        return self._sampler

    def fit(self, X: np.array, y: np.array) -> None:
        (self._x_majority, self._y_majority), (self._x_minority, self._y_minority) = divide_dataset(X, y)
        self._prepare_dataset()
        self._set_classificators_count()
        self._instantiate_fitted_classificators()

    @abstractmethod
    def predict(self, X: np.array) -> np.array:
        raise NotImplemented()

    @property
    def minority_count(self) -> int:
        return 0 if self._y_minority is None else self._y_minority.size

    @property
    def majority_count(self) -> int:
        return 0 if self._y_majority is None else self._y_majority.size

    def _set_classificators_count(self) -> None:
        self._needed_classificators_count = ceil(self.majority_count / self.minority_count)

    def _prepare_dataset(self) -> None:
        if self.majority_count % self.minority_count != 0:
            oversampling_count = self.minority_count * ceil(self.majority_count / self.minority_count)  # this will always be sligthly larger or equal majority_count
            x_majority = choices(self._x_minority, k=oversampling_count)  # majority in terms of smote. In fact, it's the
                                                                          # minority class
            y_majority = np.array([self._y_minority[0]] * oversampling_count)
            x_minority = self._x_majority[:oversampling_count, :]
            y_minority = self._y_majority[:oversampling_count]
            x = np.concatenate((x_majority, x_minority), axis=0)
            y = np.concatenate((y_majority, y_minority), axis=0)

            sampled_x, sampled_y = self._sampler.fit_resample(x, y)
            majority_indices = np.where(sampled_y == self._y_majority[0])[0][:oversampling_count]
            self._x_majority = sampled_x[majority_indices, :]
            self._y_majority = sampled_y[majority_indices]

    def _prepare_opt_ds(self, ds_size: int) -> Tuple[np.array, np.array, np.array, np.array]:
        test_size = int(round(ds_size * self._opt_size))
        train_size = ds_size - test_size
        ir = ceil(self.majority_count / self.minority_count)
        # test
        minority_count = floor(test_size / (ir + 1))
        majority_count = test_size - minority_count
        x_minority = choices(self._x_minority, k=minority_count)
        x_majority = choices(self._x_majority, k=majority_count)
        x_test = np.concatenate([x_minority, x_majority], axis=0)
        y_test = np.concatenate([
            np.array([self._y_minority[0] * minority_count]),
            np.array([self._y_majority[0] * majority_count])
        ])
        # train
        minority_count = floor(train_size / (ir + 1))
        majority_count = train_size - minority_count
        x_minority = choices(self._x_minority, k=minority_count)
        x_majority = choices(self._x_majority, k=majority_count)
        x_train = np.concatenate([x_minority, x_majority], axis=0)
        y_train = np.concatenate([
            np.array([self._y_minority[0] * minority_count]),
            np.array([self._y_majority[0] * majority_count])
        ])

        return x_train, x_test, y_train, y_test

    def _instantiate_fitted_classificators(self) -> None:
        start_index = 0
        minority_count = self._y_minority.size
        self._classifiers = []
        for _ in range(self._needed_classificators_count):
            classifier = self.__init_classifier(self._predict_proba)
            end_index = start_index + minority_count
            x_majority = self._x_majority[start_index:end_index, :]
            y_majority = self._y_majority[start_index:end_index]
            x = np.concatenate((x_majority, self._x_minority), axis=0)
            y = np.concatenate((y_majority, self._y_minority), axis=0)

            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x,
                    y,
                    test_size=self._opt_size,
                    stratify=y,
                    random_state=self._random_state
                )
            except ValueError:
                x_train, x_test, y_train, y_test = self._prepare_opt_ds(x.shape[0])

            classifier.fit(x_train, y_train)
            optimizer = ThresholdOptimizerFactory.get(self._optimization)
            optimizer.optimize(
                y_test,
                classifier.predict(x_test) if not self._predict_proba else classifier.predict_proba(x_test)[:, 1]
            )

            self._classifiers.append(OptimizedCls(classifier, optimizer))
            start_index += minority_count


class BaggFold(BaseBaggFold):
    def predict(self, X) -> np.array:
        return predict(
            classifiers=self._classifiers,
            X=X,
            voting_type=self._voting_type,
            predict_proba=self._predict_proba
        )


class BaggFoldThread(Thread):
    def __init__(self, classifiers: List[OptimizedCls], X: np.array, predict_proba: bool = False):
        super().__init__()
        self.__classifiers = classifiers
        self.__X = X
        self.predictions = None
        self.__predict_proba = predict_proba

    def run(self) -> None:
        self.predictions = predict(
            classifiers=self.__classifiers,
            X=self.__X,
            predict_proba=self.__predict_proba
        )

    @property
    def thresholds(self) -> np.array:
        return np.array([cls.optimizer.threshold for cls in self.__classifiers])


class ThreadedBaggFold(BaseBaggFold):
    def __init__(
            self,
            base_classifier_fn: Callable[[], object],
            sampler,
            optimization_type: OptimizationType = OptimizationType.ROC,
            voting_type: VotingType = VotingType.MEAN,
            optimization_ds_size: float = .1,
            max_threads: int = 30,
            predict_proba: bool = False,
            random_state: int = 100,
    ):
        super().__init__(
            base_classifier_fn=base_classifier_fn,
            sampler=sampler,
            optimization_type=optimization_type,
            voting_type=voting_type,
            optimization_ds_size=optimization_ds_size,
            predict_proba=predict_proba,
            random_state=random_state
        )
        self.__threads = []
        self.__max_threads = max_threads

    def predict(self, X: np.array) -> np.array:
        preferred_threads_count = self.__max_threads\
            if self._needed_classificators_count >= 30 \
            else self._needed_classificators_count
        self.__prepare_threads(preferred_threads_count, X)
        classifications = np.array([])
        thresholds = np.array([])
        for thread in self.__threads:
            thread.start()
        for thread in self.__threads:
            thread.join()
            classifications = np.append(classifications, thread.predictions)
            thresholds = np.concatenate((thresholds, thread.thresholds), axis=0)

        classifications = np.reshape(classifications, (self._needed_classificators_count, -1))
        mean_thresholds = np.mean(thresholds)
        voting = VotingFactory.get(voting=self._voting_type, predict_proba=self._predict_proba)

        return voting.vote(classifications, mean_thresholds)

    def __prepare_threads(self, threads_count: int, X: np.array) -> int:
        classifiers_per_thread = ceil(self._needed_classificators_count / threads_count)
        start_index = 0
        self.__threads = []
        classifiers_left = self._needed_classificators_count
        for _ in range(threads_count):
            end_index = start_index + classifiers_per_thread
            classifiers = self._classifiers[start_index:end_index]
            thread = BaggFoldThread(classifiers, X, self._predict_proba)
            self.__threads.append(thread)
            start_index += classifiers_per_thread
            classifiers_left -= len(classifiers)
            if classifiers_left == 0:
                break

        return len(self.__threads)
