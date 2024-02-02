from __future__ import annotations
from typing import Union, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils import divide_dataset


RawDataset = Tuple[np.ndarray, np.ndarray]
RAW_DIR = Path(__file__).parent.parent / "raw"


class DatasetFactory:
    @staticmethod
    def _load_raw_dataset(dataset_name: str) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / dataset_name, comment='@', header=None, delimiter=',')

    @staticmethod
    def _get_dataset(dataset_name: str, raw_dataset: bool = False) -> Union[Dataset, RawDataset]:
        le = LabelEncoder()
        df = DatasetFactory._load_raw_dataset(dataset_name)
        X, y = df.iloc[:, :-1].values, le.fit_transform(df.iloc[:, -1].values)

        if raw_dataset:
            return X, y

        return Dataset(X=X, y=y, name=dataset_name)

    @staticmethod
    def ecoli1() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('ecoli1.dat')

    @staticmethod
    def ecoli4() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('ecoli4.dat')

    @staticmethod
    def yeast1() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('yeast1.dat')

    @staticmethod
    def yeast3() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('yeast3.dat')

    @staticmethod
    def yeast6() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('yeast6.dat')

    @staticmethod
    def new_thyroid1() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('new-thyroid1.dat')

    @staticmethod
    def iris0() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('iris0.dat')

    @staticmethod
    def glass2() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('glass2.dat')

    @staticmethod
    def abalone19() -> Dataset:
        """ Dataset not scaled """

        ds_name = 'abalone19.dat'
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(drop='first'), [0])],
            remainder='passthrough'
        )
        X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
        X = np.array(ct.fit_transform(X))

        return Dataset(X=X, y=y, name=ds_name, categorical_columns=np.array([0, 1]))

    @staticmethod
    def kr_vs_k() -> Dataset:
        """ Dataset with only categorical data """

        ds_name = 'kr-vs-k-zero_vs_eight.dat'
        categorical_columns = np.arange(0, 35)
        ct = ColumnTransformer(
            transformers=[
                ('cols', OneHotEncoder(drop='first'), [0, 2, 4]),
                ('rows', OneHotEncoder(drop='first'), [1, 3, 5]),
            ],
            remainder='passthrough'
        )
        X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
        X = np.array(ct.fit_transform(X).toarray())

        return Dataset(X=X, y=y, name=ds_name, categorical_columns=categorical_columns)

    @staticmethod
    def dermatology() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('dermatology-6.dat')

    @staticmethod
    def get_all() -> List[Dataset]:
        return [
            DatasetFactory.yeast1(),
            DatasetFactory.yeast3(),
            DatasetFactory.yeast6(),
            DatasetFactory.new_thyroid1(),
            DatasetFactory.iris0(),
            DatasetFactory.glass2(),
            DatasetFactory.ecoli1(),
            DatasetFactory.ecoli4(),
            DatasetFactory.abalone19(),
            DatasetFactory.kr_vs_k(),
            DatasetFactory.dermatology()
        ]


class Dataset:
    def __init__(
            self,
            X: np.array,
            y: np.array,
            name: str,
            categorical_columns: Optional[np.array] = None
    ):
        self._X = X
        self._y = y
        self._name = name
        self._cat_cols = np.array([]) if categorical_columns is None else categorical_columns

    @property
    def X(self) -> np.array:
        return self._X

    @property
    def y(self) -> np.array:
        return self._y

    @property
    def name(self) -> str:
        return self._name

    @property
    def categorical_columns(self) -> np.array:
        return self._cat_cols

    @property
    def ir(self) -> float:
        (x_maj, _), (x_min, _) = divide_dataset(self.X, self.y)

        return len(x_maj) / len(x_min)
