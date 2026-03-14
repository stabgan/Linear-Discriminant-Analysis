# Linear Discriminant Analysis (LDA)
# Dimensionality reduction + classification on the Wine dataset

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------------------------------------------------------------------
# Resolve dataset path relative to this script so it works from any cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_SCRIPT_DIR, "Wine.csv")

# --- Visualization helpers ------------------------------------------------
COLORS = ("red", "green", "blue")
CMAP = ListedColormap(COLORS)


def plot_decision_boundary(classifier, X_set, y_set, title):
    """Draw a 2-D decision-boundary plot for *classifier* over *X_set*."""
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
    )
    plt.contourf(
        X1,
        X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
            X1.shape
        ),
        alpha=0.75,
        cmap=CMAP,
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=COLORS[i],
            edgecolors="k",
            label=int(j),
        )
    plt.title(title)
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend()
    plt.show()


# --- Main -----------------------------------------------------------------

def main():
    # Load dataset
    dataset = pd.read_csv(_DATA_PATH)
    X = dataset.iloc[:, 0:13].values
    y = dataset.iloc[:, 13].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Apply LDA (project onto 2 discriminant components)
    lda = LDA(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    # Logistic Regression classifier
    classifier = LogisticRegression(
        solver="lbfgs", max_iter=200, random_state=0
    )
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {acc:.2%}")

    # Decision-boundary plots
    plot_decision_boundary(
        classifier, X_train, y_train, "Logistic Regression (Training set)"
    )
    plot_decision_boundary(
        classifier, X_test, y_test, "Logistic Regression (Test set)"
    )


if __name__ == "__main__":
    main()
