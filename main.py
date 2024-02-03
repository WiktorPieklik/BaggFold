from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from simple_chalk import green, cyan, magenta, yellow

from src.baggfold import ThreadedBaggFold
from src.smote import (
    CenterPointSMOTE as CPSmote,
    InnerOuterSMOTE as IOSmote
)
from src.datasets import DatasetFactory
from src.utils import f1_measure, VotingType


if __name__ == "__main__":
    base_classifiers = {
        'SVM': lambda proba=False: SVC(probability=proba, random_state=10),
        'CART': lambda proba=False: DecisionTreeClassifier(random_state=11),
        'XGB': lambda proba=False: XGBClassifier()
    }
    smotes = {
        'io-smote': IOSmote(n=5),
        'cp-smote': CPSmote(n=5, random_state=12)
    }
    voting_mechanisms = {
        'avg voting': VotingType.MEAN,
        'majority voting': VotingType.MAJORITY
    }
    datasets = [DatasetFactory.iris0(), DatasetFactory.yeast3(), DatasetFactory.ecoli1()]

    for dataset in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=.1,
            stratify=dataset.y,
            random_state=21
        )
        for base_cls_name, cls_fn in base_classifiers.items():
            for smote_name, smote in smotes.items():
                for voting_name, voting in voting_mechanisms.items():
                    model_name = cyan(f"BaggFold({base_cls_name}) + {smote_name} + {voting_name}")
                    baggfold = ThreadedBaggFold(
                        base_classifier_fn=cls_fn,
                        sampler=smote,
                        voting_type=voting,
                        random_state=37
                    )
                    baggfold.fit(X_train, y_train)
                    score, predict_time_ns = f1_measure(baggfold, X_test, y_test)
                    score_str = green(f"{score:.2f}")
                    predict_time_str = magenta(f"{predict_time_ns / 10e9:.3f}")

                    print(f"F1 score: {score_str} in {predict_time_str} seconds on {yellow(dataset.name)} for {model_name}")
                    print("##########")
