from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from simple_chalk import green, cyan, magenta, yellow
from imblearn.over_sampling import SMOTE

from src.baggfold import ThreadedBaggFold
from src.datasets import DatasetFactory
from src.utils import f1_measure, g_mean, auc_score, brier_score, log_loss, VotingType


if __name__ == "__main__":
    baggfolds = {
        'BaggFold(CART) + SMOTE + majority voting': ThreadedBaggFold(
            base_classifier_fn=lambda proba=False: DecisionTreeClassifier(random_state=11),
            sampler=SMOTE(sampling_strategy=1.0, random_state=21),
            voting_type=VotingType.MAJORITY,
            random_state=37
        ),
        'BaggFold(XGB) + SMOTE + majority voting': ThreadedBaggFold(
            base_classifier_fn=lambda proba=False: XGBClassifier(),
            sampler=SMOTE(sampling_strategy=1.0, random_state=21),
            voting_type=VotingType.MAJORITY,
            random_state=37
        ),
        'BaggFold(SVM) + SMOTE + average voting': ThreadedBaggFold(
            base_classifier_fn=lambda proba: SVC(probability=proba, random_state=21),
            sampler=SMOTE(sampling_strategy=1.0, random_state=21),
            voting_type=VotingType.MEAN,
            random_state=37,
            predict_proba=True
        )
    }
    datasets = [
        DatasetFactory.iris0(),
        DatasetFactory.yeast6(),
        DatasetFactory.ecoli1(),
        DatasetFactory.glass4(),
        DatasetFactory.abalone19(),
        DatasetFactory.kr_vs_k(),
        DatasetFactory.wine_quality(),
    ]

    for dataset in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=.2,
            stratify=dataset.y,
            random_state=21
        )
        for name, baggfold in baggfolds.items():
            if "SVM" in name:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                baggfold.fit(X_train_scaled, y_train)
                f1, predict_time_ns = f1_measure(baggfold, X_test_scaled, y_test)
                g, _ = g_mean(baggfold, X_test_scaled, y_test)
                auc, _ = auc_score(baggfold, X_test_scaled, y_test)
                brier, _ = brier_score(baggfold, X_test_scaled, y_test)
                log, _ = log_loss(baggfold, X_test_scaled, y_test)
                predict_time_str = magenta(f"{predict_time_ns / 10e9:.4f}")
            else:
                baggfold.fit(X_train, y_train)
                f1, predict_time_ns = f1_measure(baggfold, X_test, y_test)
                g, _ = g_mean(baggfold, X_test, y_test)
                auc, _ = auc_score(baggfold, X_test, y_test)
                brier, _ = brier_score(baggfold, X_test, y_test)
                log, _ = log_loss(baggfold, X_test, y_test)
                predict_time_str = magenta(f"{predict_time_ns / 10e9:.4f}")

            print(f"G-mean: {g:.4f}, F1 score: {f1:.4f}, AUC: {auc:.4f}, Brier score: {brier:.4f}, Log loss: {log:.4f} in {predict_time_str} seconds on {yellow(dataset.pretty_name)} for {name}")
