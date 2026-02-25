from __future__ import annotations
from typing import Union, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.utils import divide_dataset, set_majority_as_one


RawDataset = Tuple[np.ndarray, np.ndarray]
RAW_DIR = Path(__file__).parent.parent / "raw"


class DatasetFactory:
    @staticmethod
    def _load_raw_dataset(dataset_name: str) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / dataset_name, comment='@', header=None, delimiter=',')

    @staticmethod
    def _get_dataset(dataset_name: str, pretty_name: Optional[str] = None, raw_dataset: bool = False) -> Union[Dataset, RawDataset]:
        le = LabelEncoder()
        df = DatasetFactory._load_raw_dataset(dataset_name)
        X, y = df.iloc[:, :-1].values, le.fit_transform(df.iloc[:, -1].values)
        (X_majority, y_majority), (X_minority, y_minority) = divide_dataset(X, y)
        y_majority, y_minority = set_majority_as_one(y_majority, y_minority)
        X = np.concatenate((X_majority, X_minority), axis=0)
        y = np.concatenate((y_majority, y_minority), axis=0)

        if raw_dataset:
            return X, y

        return Dataset(X=X, y=y, name=dataset_name, pretty_name=pretty_name)

    @staticmethod
    def ecoli1() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('ecoli1.dat')

    @staticmethod
    def ecoli3() -> Dataset:
        """ Dataset already scaled """

        return DatasetFactory._get_dataset('ecoli3.dat')

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
    def glass4() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('glass4.dat')

    @staticmethod
    def glass6() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('glass6.dat')

    @staticmethod
    def abalone19(ordinal_encoding: bool = True) -> Dataset:
        """ Dataset not scaled """

        ds_name = 'abalone19.dat'

        if ordinal_encoding:
            ct = ColumnTransformer(
                transformers=[('encoder', OrdinalEncoder(), [0])],
                remainder='passthrough'
            )
            X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
            X = np.array(ct.fit_transform(X))

            return Dataset(X=X, y=y, name=ds_name)
        else:
            ct = ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(drop='first'), [0])],
                remainder='passthrough'
            )
            X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
            X = np.array(ct.fit_transform(X))

            return Dataset(X=X, y=y, name=ds_name, categorical_columns=np.array([0, 1]))

    @staticmethod
    def kr_vs_k(ordinal_encoding: bool = True) -> Dataset:
        """ Dataset with only categorical data """

        ds_name = 'kr-vs-k-zero_vs_eight.dat'

        if ordinal_encoding:
            ct = ColumnTransformer(
                transformers=[
                    ('cols', OrdinalEncoder(), [0, 2, 4]),
                    ('rows', OrdinalEncoder(), [1, 3, 5])
                ],
                remainder='passthrough'
            )
            X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
            X = np.array(ct.fit_transform(X))

            return Dataset(X=X, y=y, name=ds_name, pretty_name='kr-vs-k')
        else:
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

            return Dataset(X=X, y=y, name=ds_name, pretty_name='kr-vs-k', categorical_columns=categorical_columns)

    @staticmethod
    def adult(ordinal_encoding: bool = True) -> Dataset:
        """ Dataset not scaled """

        ds_name = 'adult.csv'

        if ordinal_encoding:
            ct = ColumnTransformer(
                transformers=[
                    ('education', OrdinalEncoder(), [2]),
                    ('marital_status', OrdinalEncoder(), [4]),
                    ('relationship', OrdinalEncoder(), [5]),
                    ('race', OrdinalEncoder(), [6]),
                    ('sex', OrdinalEncoder(), [7]),
                ],
                remainder='passthrough'
            )
            X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
            X = np.array(ct.fit_transform(X))

            return Dataset(X=X, y=y, name=ds_name)
        else:
            ct = ColumnTransformer(
                transformers=[
                    ('education', OneHotEncoder(drop='first'), [2]),  # 15
                    ('marital_status', OneHotEncoder(drop='first'), [4]),  # 6
                    ('relationship', OneHotEncoder(drop='first'), [5]),  # 5
                    ('race', OneHotEncoder(drop='first'), [6]),  # 4
                    ('sex', OneHotEncoder(drop='first'), [7]),  # 1
                ],
                remainder='passthrough'
            )
            categorical_columns = np.array([
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28,
                29, 30, 31, 32,
                33
            ])
            X, y = DatasetFactory._get_dataset(ds_name, raw_dataset=True)
            X = np.array(ct.fit_transform(X).toarray())

            return Dataset(X=X, y=y, name=ds_name, categorical_columns=categorical_columns)

    @staticmethod
    def credit_card() -> Dataset:
        ds_name = 'creditcard.csv'

        return DatasetFactory._get_dataset(ds_name, pretty_name='credit card fraud')

    @staticmethod
    def dermatology() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('dermatology-6.dat')

    @staticmethod
    def page_blocks() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('page-blocks0.dat', pretty_name='page blocks')

    @staticmethod
    def segment() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('segment0.dat', pretty_name='segment')

    @staticmethod
    def vowel() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('vowel0.dat', pretty_name='vowel')

    @staticmethod
    def wine_quality() -> Dataset:
        """ Dataset not scaled """

        return DatasetFactory._get_dataset('winequality-red-4.dat', pretty_name='wine quality')

    @staticmethod
    def get_all(ordinal_encoding: bool = True) -> List[Dataset]:
        return [
            DatasetFactory.yeast1(),
            DatasetFactory.yeast3(),
            DatasetFactory.yeast6(),
            DatasetFactory.new_thyroid1(),
            DatasetFactory.iris0(),
            DatasetFactory.glass2(),
            DatasetFactory.glass4(),
            DatasetFactory.glass6(),
            DatasetFactory.ecoli1(),
            DatasetFactory.ecoli3(),
            DatasetFactory.ecoli4(),
            DatasetFactory.abalone19(ordinal_encoding),
            DatasetFactory.kr_vs_k(ordinal_encoding),
            DatasetFactory.dermatology(),
            DatasetFactory.page_blocks(),
            DatasetFactory.segment(),
            DatasetFactory.vowel(),
            DatasetFactory.wine_quality(),
            DatasetFactory.adult(ordinal_encoding),
            DatasetFactory.credit_card()
        ]


class Dataset:
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            name: str,
            pretty_name: Optional[str] = None,
            categorical_columns: Optional[np.array] = None
    ):
        self._X = X
        self._y = y
        self._name = name
        self._pretty_name = pretty_name
        self._cat_cols = np.array([]) if categorical_columns is None else categorical_columns

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name if self._pretty_name is not None else self._name[:-4]

    @property
    def categorical_columns(self) -> np.ndarray:
        return self._cat_cols

    @property
    def ir(self) -> float:
        (x_maj, _), (x_min, _) = divide_dataset(self.X, self.y)

        return len(x_maj) / len(x_min)

    @property
    def imbalanced_props(self) -> Tuple[int, int, int]:
        (x_maj, _), (x_min, _) = divide_dataset(self.X, self.y)

        return len(self.y), len(x_min), len(x_maj)
