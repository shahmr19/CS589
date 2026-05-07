import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def load_parkinsons():
    df = pd.read_csv("../parkinsons.csv")
    X = df.drop(columns=["Diagnosis"]).values
    y = df["Diagnosis"].values
    return X, y


def train_gaussian_nb(X_train, y_train, epsilon):
    labels = np.unique(y_train)
    model = {}

    for label in labels:
        X_c = X_train[y_train == label]

        model[label] = {
            "prior": len(X_c) / len(X_train),
            "mean": np.mean(X_c, axis=0),
            "var": np.var(X_c, axis=0) + epsilon
        }
    return model

def predict_one(model, x):
    scores = {}

    for label, params in model.items():
        prior = np.log(params["prior"])
        mean = params["mean"]
        var = params["var"]

        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((x - mean) ** 2) / var
        )

        scores[label] = prior + log_likelihood

    return max(scores, key=scores.get)


def predict(model, X):
    return np.array([predict_one(model, x) for x in X])


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


def evaluate_nb(X, y, epsilon, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

    accs = []
    f1s = []

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Normalize inside fold
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        model = train_gaussian_nb(X_train, y_train, epsilon)
        preds = predict(model, X_test)

        accs.append(accuracy_score(y_test, preds))
        f1s.append(macro_f1_score(y_test, preds))

    return {
        "accuracy": np.mean(accs),
        "f1": np.mean(f1s),
        "accuracy_std": np.std(accs),
        "f1_std": np.std(f1s)
    }
def run_experiments(X, y, output_prefix):
    epsilons = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1]

    accs = []
    acc_stds = []
    f1s = []
    f1_stds = []

    print("Gaussian Naive Bayes on Parkinsons")
    print("epsilon\taccuracy\tf1")

    for epsilon in epsilons:
        results = evaluate_nb(X, y, epsilon, n_splits=10)

        accs.append(results["accuracy"])
        acc_stds.append(results["accuracy_std"])
        f1s.append(results["f1"])
        f1_stds.append(results["f1_std"])

        print(
            f"{epsilon}\t"
            f"{results['accuracy']:.4f}\t\t"
            f"{results['f1']:.4f}"
        )

    plt.figure()
    plt.errorbar(epsilons, accs, yerr=acc_stds, fmt="o-")
    plt.xscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("Accuracy")
    plt.title("Parkinsons Gaussian NB Accuracy vs epsilon")
    plt.grid(True)
    plt.savefig(f"{output_prefix}_nb_accuracy.png")
    plt.close()

    plt.figure()
    plt.errorbar(epsilons, f1s, yerr=f1_stds, fmt="o-")
    plt.xscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("Macro F1-score")
    plt.title("Gaussian NB F1 vs epsilon")
    plt.grid(True)
    plt.savefig(f"{output_prefix}_nb_f1.png")
    plt.close()


def main():

    X_parkinsons, y_parkinsons = load_parkinsons()
    run_experiments( X_parkinsons, y_parkinsons, "parkinsons")


if __name__ == "__main__":
    main()
