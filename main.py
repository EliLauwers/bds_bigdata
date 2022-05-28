import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime as datetime
import os
import matplotlib.pyplot as plt
import pandas as pd

random_state = 1234
np.random.seed(random_state)


def create_results():
    save_path = "run_times.psv"
    data = pd.read_csv("SFinGe_VQAndPert.csv")

    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=1000, stratify=y, random_state=random_state
    )

    classifiers = [
        ("KNN", KNeighborsClassifier(n_neighbors=1)),
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                criterion="entropy",
                max_depth=7,
                max_features="log2",
                n_estimators=200,
                random_state=random_state,
            ),
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                C=1, penalty="l2", random_state=random_state, max_iter=10000
            ),
        ),
    ]
    max_power = 20
    for p in range(3, max_power + 1):
        if 2 ** p > len(y_train_pool):
            index_vec = np.random.choice(X_train_pool.index, size=(2 ** p) + 100)
        else:
            index_vec = X_train_pool.index
        X_train, _, y_train, _ = train_test_split(
            X_train_pool.loc[index_vec],
            y_train_pool.loc[index_vec],
            train_size=2 ** p,
            stratify=y_train_pool.loc[index_vec],
            random_state=random_state,
        )

        for clf_name, clf in classifiers:
            print(f"\rnow at {p} with max {max_power}", end=" " * 10)
            fit_start_time = datetime.now()
            model = clf.fit(X_train, y_train)
            predictions = model.predict(X_test)
            query_time = (datetime.now() - fit_start_time).total_seconds()
            # Measure run times
            fit_start_time = datetime.now()
            model = clf.fit(X_train, y_train)
            construction_time = (datetime.now() - fit_start_time).total_seconds()
            predict_start_time = datetime.now()
            y_scores = model.predict(X_test)
            query_time = (datetime.now() - predict_start_time).total_seconds()
            # We only store ‘p‘ since that is enough to reconstruct the training set size.
            result = {
                "p": p,
                "classifier": clf_name,
                "construction_time": construction_time,
                # Divide the query time by 1000 to obtain the query time per instance.
                "query_time": query_time / len(X_test),
            }
            # We store each line of our table separately. This allows us to stop the process
            # early and still have results. Alternatively, you could first create
            # a dataframe with all the results and store this at the end.
            result = pd.DataFrame(result, index=[0])
            # The following lines ensure that we don’t overwrite the entire existing file,
            # but only those lines with the same ‘p‘ and ‘classifier‘.
            index_cols = ["p", "classifier"]
            result = result.set_index(index_cols)
            if os.path.exists(save_path):
                old = pd.read_csv(save_path, sep="|", index_col=index_cols)
                result = result.combine_first(old)
            result.to_csv(save_path, sep="|", index=True)


def process_results():
    run_times = pd.read_csv(
        "run_times.psv",
        sep="|",
    )
    # Pivot to obtain columns that correspond to the different classifiers
    construction_time = run_times.pivot(
        index="p", columns="classifier", values="construction_time"
    )
    construction_time.plot()
    # Make the y-axis logarithmic. Since we stored training set size in terms of the
    # exponent ‘p‘, the x-axis is already logarithmic
    axes = plt.gca()
    axes.set_yscale("log", base=10)
    labels = axes.get_xticks()
    # It's ok to just label the x-axis as p, but we are going to be a bit more fancy here
    # and change the labels back to 2ˆp, using the subset of latex enabled by matplotlib
    axes.set_xticklabels(
        map(lambda x: "$\mathregular{{2ˆ{{{}}}}}$".format(int(x)), labels)
    )
    axes.set_xlabel("Training set size")
    axes.set_ylabel("Construction time (s)")
    plt.savefig("construction_times.pdf")
    # And everything again for query times. You could also use a loop to not repeat most of this code.
    query_times = run_times.pivot(index="p", columns="classifier", values="query_time")
    query_times.plot()
    axes = plt.gca()
    axes.set_yscale("log", base=10)
    labels = axes.get_xticks()
    axes.set_xticklabels(
        map(lambda x: "$\mathregular{{2ˆ{{{}}}}}$".format(int(x)), labels)
    )
    axes.set_xlabel("Training set size")
    axes.set_ylabel("Query time (s) per instance")
    plt.savefig("query_times.pdf")


if __name__ == "__main__":
    create_results()
    process_results()
    print("done")
