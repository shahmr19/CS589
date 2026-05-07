import math
import random
from sklearn import datasets
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, leaf=False, label=None, split=None, threshold=None, numeric=False):
        self.leaf = leaf
        self.label = label
        self.split = split
        self.children = {}
        self.threshold = threshold
        self.numeric = numeric
        self.left = None
        self.right = None

def setup():
    dataset = datasets.load_digits()
    X = dataset.images.reshape((len(dataset.images), -1))
    y = dataset.target

    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    return X, y

def major_label(y):
    return y.mode().iloc[0]

def entropy(x):
    count = x.value_counts()
    prob = count/count.sum()
    return -np.sum(prob*np.log2(prob))

def numeric_attr(X, attr):
    if str(attr).endswith("_num"):
        return True
    if str(attr).endswith("_cat"):
        return False
    return pd.api.types.is_numeric_dtype(X[attr])

def bootstrap(X, y, seed=123):
    rng = np.random.default_rng(seed)
    n = len(X)
    ind = rng.integers(0, n, size=n)
    return (X.iloc[ind].reset_index(drop=True), y.iloc[ind].reset_index(drop=True))

def make_stratified_folds(X, y, k=5, seed=123):
    rng = np.random.default_rng(seed)
    label_to_ind = {}
    for label in y.unique():
        idx = y[y == label].index.to_numpy().copy()
        rng.shuffle(idx)
        label_to_ind[label] = np.array_split(idx, k)
    folds = []
    for i in range(k):
        fold_ind = []
        for label in label_to_ind:
            fold_ind.extend(label_to_ind[label][i].tolist())
        rng.shuffle(fold_ind)
        fold_ind = np.array(fold_ind)

        X_fold = X.loc[fold_ind].reset_index(drop=True)
        y_fold = y.loc[fold_ind].reset_index(drop=True)
        folds.append((X_fold, y_fold))
    
    return folds

def combine(folds, test_idx):
    X_test, y_test = folds[test_idx]

    X_train = []
    y_train =[]
    for i, (X_fold, y_fold) in enumerate(folds):
        if i != test_idx:
            X_train.append(X_fold)
            y_train.append(y_fold)
    X_train_final = pd.concat(X_train, ignore_index=True)
    y_train_final = pd.concat(y_train, ignore_index=True)
    return X_train_final, y_train_final, X_test, y_test




def info_gain_cat(X, y, attribute):
    base_ent = entropy(y)
    cond_ent = 0
    for v,d in X.groupby(attribute):
        y_d = y.loc[d.index]
        weight = len(y_d)/len(y)
        cond_ent += weight * entropy(y_d)
    gain = base_ent - cond_ent
    return gain

def info_gain_numeric(X, y, attribute):
    threshold = X[attribute].mean()
    base_ent = entropy(y)
    l_mask = X[attribute] <= threshold
    r_mask = X[attribute] > threshold
    y_left = y.loc[X[l_mask].index]
    y_right = y.loc[X[r_mask].index]
    cond_ent = 0
    if len(y_left) > 0:
        cond_ent += (len(y_left)/len(y)) * entropy(y_left)
    if len(y_right) > 0:
        cond_ent += (len(y_right)/len(y)) * entropy(y_right)
    gain = base_ent - cond_ent
    return gain, threshold

def best_split(X, y, attributes, m_try, rng):
    if len(attributes) == 0:
        return None
    
    m = min(m_try, len(attributes))
    sample_attrs = rng.sample(attributes, m)
    best_attr = None
    best_gain = -np.inf
    best_threshold = None
    best_numeric = False

    for attr in sample_attrs:
        if numeric_attr(X, attr):
            gain, threshold = info_gain_numeric(X, y, attr)
            numeric = True
        else:
            gain = info_gain_cat(X, y, attr)
            threshold=None
            numeric = False
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
            best_threshold = threshold
            best_numeric = numeric
    
    return best_attr, best_threshold, best_numeric, best_gain

def DT(X, y, attributes, m_try, max_depth, min_split=2, min_gain=1e-9, depth=0, rng=None):
    if rng is None:
        rng = random.Random(0)
    node  = Node()
    node.label = major_label(y)
    if y.nunique() == 1:
        node.leaf = True
        node.label = y.iloc[0]
        return node
    if not attributes:
        node.leaf=True
        return node
    if len(X)<min_split:
        node.leaf = True
        return node
    if max_depth is not None and depth >= max_depth:
        node.leaf = True
        return node
    
    split = best_split(X, y, attributes, m_try, rng)
    if split[0] is None:
        node.leaf = True
        return node
    
    best_a, threshold, best_numeric, best_gain = split

    if best_gain < min_gain:
        node.leaf = True
        return node
    node.leaf=False
    node.split=best_a
    node.threshold = threshold
    node.numeric = best_numeric
    remaining_attributes = [a for a in attributes if a != best_a]
    if best_numeric:
        l_mask = X[best_a] <= threshold
        r_mask = X[best_a] > threshold
        X_left = X[l_mask]
        X_right = X[r_mask]
        y_left = y.loc[X_left.index]
        y_right = y.loc[X_right.index]

        if len(y_left) == 0 or len(y_right) == 0:
            node.leaf = True
            node.left = None
            node.right =None
            return node
        
        node.left = DT(X_left, y_left, attributes, m_try, max_depth=max_depth, min_split=min_split, min_gain=min_gain, depth=depth+1, rng=rng)
        node.right = DT(X_right, y_right, attributes, m_try, max_depth=max_depth, min_split=min_split, min_gain=min_gain, depth=depth+1, rng=rng)
    else:
        for v,d in X.groupby(best_a):
            X_v = d
            y_v = y.loc[d.index]
            child = DT(X_v, y_v, remaining_attributes, m_try, max_depth=max_depth, min_split=min_split, min_gain=min_gain, depth=depth+1, rng=rng)
            node.children[v] = child
    return node

def predict(node, x):
    if node.leaf:
        return node.label
    if node.numeric:
        if x[node.split] <= node.threshold:
            if node.left is None:
                return node.label
            return predict(node.left, x)
        else:
            if node.right is None:
                return node.label
            return predict(node.right, x)
    
    value = x[node.split]
    child = node.children.get(value)
    if child is None:
        return node.label
    return predict(child, x)

def forest(X_train, y_train, n_trees, m_try, max_depth=None, min_split=2, min_gain=1e-9, seed=123):
    f=[]
    attributes = list(X_train.columns)

    for t in range(n_trees):
        t_seed = seed + 1000 * (t + 1)
        rng = random.Random(t_seed)
        X_boot, y_boot = bootstrap(X_train, y_train, seed=t_seed)
        tree = DT(X_boot, y_boot, attributes=attributes, m_try=m_try, max_depth=max_depth, min_split=min_split, min_gain=min_gain, depth=0, rng=rng)
        f.append(tree)
    return f

def predict_forest(f, x):
    votes = [predict(tree, x) for tree in f]
    return Counter(votes).most_common(1)[0][0]

def metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
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

    return {
        "accuracy": accuracy,
        "f1": np.mean(f1_scores)
    }

def evaluate(X, y, n_trees, k_folds=10, m_try=None, max_depth=5, min_split=5, min_gain=1e-9, seed=123):
    if m_try is None:
        m_try = max(1, int(math.sqrt(X.shape[1])))
    folds = make_stratified_folds(X, y, k=k_folds, seed=seed)
    fold_metrics = []
    for i in range(k_folds):
        X_train, y_train, X_test, y_test  = combine(folds, i)
        f = forest(X_train, y_train, n_trees=n_trees, m_try=m_try, max_depth=max_depth, min_split=min_split, min_gain=min_gain, seed=seed+i)
        preds=[]
        for _, row in X_test.iterrows():
            preds.append(predict_forest(f, row))

        m = metrics(y_test.values, np.array(preds))
        fold_metrics.append(m)
    result = {}
    for metric in ["accuracy", "f1"]:
        result[metric] = np.mean([fm[metric] for fm in fold_metrics])

    return result

def experiments(dataset, X, y, ntree_vals, op_prefix, max_depth=5, min_split=5, min_gain=1e-9, seed=123):
    results = {
        "accuracy":[],
        "f1": []
    }

    m_try = max(1, int(math.sqrt(X.shape[1])))

    for n_trees in ntree_vals:
        res = evaluate(X, y, n_trees=n_trees, k_folds=10, m_try=m_try, max_depth=max_depth, min_split=min_split, min_gain=min_gain, seed=seed)
        for key in results:
            results[key].append(res[key])
    for metric in ["accuracy", "f1"]:
        plt.figure()
        plt.plot(ntree_vals, results[metric], marker="o")
        plt.xlabel("ntree")
        plt.ylabel(metric.capitalize())
        plt.title(f"{dataset}: {metric.capitalize()} vs ntree")
        plt.grid(True)
        plt.savefig(f"{op_prefix}_{metric}.png", bbox_inches="tight")
        plt.close()

    return results        

def main():
    ntree_vals = [1, 2, 3, 5, 7, 10]

    max_depth = 5
    min_split = 5
    min_gain = 1e-9

    X_digits, y_digits = setup()

    experiments(
        dataset="Digits",
        X=X_digits,
        y=y_digits,
        ntree_vals=ntree_vals,
        op_prefix="digits_tree",
        max_depth=max_depth,
        min_split=min_split,
        min_gain=min_gain,
        seed=123
    )

if __name__ == "__main__":
    main()





