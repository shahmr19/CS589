import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def euclidean(x1, x2):
    return np.sqrt(np.sum((x2-x1)**2))

def setup():
    dataset = sns.load_dataset("penguins")
    dataset = dataset.dropna()
    for col in dataset.columns:
        if dataset[col].dtype == object:
            dataset[col] = pd.factorize(dataset[col])[0]
    X = dataset.drop(columns=["species"]).values
    y = dataset["species"].values
    return X, y

def kNN(x, X_train, y_train, k, distance):
    arr = []
    for i in range(len(X_train)):
        arr.append(distance(x, X_train[i]))

    arr = np.array(arr)
    arr_index = np.argsort(arr)[:k]
    arr_labels = y_train[arr_index]
    return Counter(arr_labels).most_common(1)[0][0]

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def macro_f1_score(y_true, y_pred):
    labels = np.unique(y_true)
    f1_scores = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        f1_scores.append(f1)

    return np.mean(f1_scores)

def runKnn(k, distance, n_splits=10):
    X, y = setup()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

    train_accs = []
    test_accs = []
    train_f1s = []
    test_f1s = []

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Normalize inside each fold
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        train_predictions = []
        for x in X_train:
            train_predictions.append(kNN(x, X_train, y_train, k, distance))
        train_predictions = np.array(train_predictions)

        test_predictions = []
        for x in X_test:
            test_predictions.append(kNN(x, X_train, y_train, k, distance))
        test_predictions = np.array(test_predictions)

        train_accs.append(accuracy_score(y_train, train_predictions))
        test_accs.append(accuracy_score(y_test, test_predictions))

        train_f1s.append(macro_f1_score(y_train, train_predictions))
        test_f1s.append(macro_f1_score(y_test, test_predictions))

    return {
        "train_acc_mean": np.mean(train_accs),
        "train_acc_std": np.std(train_accs),
        "test_acc_mean": np.mean(test_accs),
        "test_acc_std": np.std(test_accs),
        "train_f1_mean": np.mean(train_f1s),
        "train_f1_std": np.std(train_f1s),
        "test_f1_mean": np.mean(test_f1s),
        "test_f1_std": np.std(test_f1s),
    }


def main():
    k_values = [5, 11, 15, 21, 25, 31, 35, 41, 45]

    train_acc_mean = []
    test_acc_mean = []
    train_acc_std = []
    test_acc_std = []

    train_f1_mean = []
    test_f1_mean = []
    train_f1_std = []
    test_f1_std = []

    for k in k_values:
        results = runKnn(k, euclidean, n_splits=10)

        train_acc_mean.append(results["train_acc_mean"])
        train_acc_std.append(results["train_acc_std"])
        test_acc_mean.append(results["test_acc_mean"])
        test_acc_std.append(results["test_acc_std"])

        train_f1_mean.append(results["train_f1_mean"])
        train_f1_std.append(results["train_f1_std"])
        test_f1_mean.append(results["test_f1_mean"])
        test_f1_std.append(results["test_f1_std"])

        print(
            f"k={k}: "
            f"test acc={results['test_acc_mean']:.4f}, "
            f"test f1={results['test_f1_mean']:.4f}"
        )

    # Accuracy graph
    plt.figure()
    plt.errorbar(k_values, test_acc_mean, yerr=test_acc_std, fmt='o-')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Penguin k-NN Accuracy vs k using 10-Fold Stratified CV")
    plt.grid(True)
    plt.savefig("penguin_knn_accuracy_cv.png")
    plt.close()

    # F1 graph
    plt.figure()
    plt.errorbar(k_values, test_f1_mean, yerr=test_f1_std, fmt='o-')
    plt.xlabel("k")
    plt.ylabel("Macro F1-score")
    plt.title("penguin k-NN Macro F1 vs k using 10-Fold Stratified CV")
    plt.grid(True)
    plt.savefig("penguin_knn_f1_cv.png")
    plt.close()

if __name__ == "__main__":
    main()




