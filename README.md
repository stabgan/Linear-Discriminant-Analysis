# 📊 Linear Discriminant Analysis (LDA)

Dimensionality reduction and multi-class classification on the UCI Wine dataset using Linear Discriminant Analysis.

## What It Does

Reduces 13 chemical features of wine samples down to 2 linear discriminants using LDA, then trains a classifier to predict customer segments — with decision-boundary visualizations for both training and test sets.

### Methodology

1. Load the Wine dataset (178 samples, 13 features, 3 classes)
2. 80/20 train/test split + standard feature scaling
3. Apply LDA → project onto 2 discriminant components
4. Train a classifier (Logistic Regression in Python, SVM in R)
5. Evaluate with a confusion matrix
6. Plot 2D decision boundaries for train and test sets

## Dataset

**Wine.csv** — 178 samples across 3 customer segments, with 13 chemical analysis features (Alcohol, Malic Acid, Ash, Magnesium, Phenols, Flavanoids, etc.). Based on the [UCI Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine).

## Quick Start

### Python

```bash
pip install numpy matplotlib pandas scikit-learn
python lda.py
```

### R

```r
install.packages(c("caTools", "MASS", "e1071", "ElemStatLearn"))
source("lda.R")
```

## Dependencies

### Python

```
numpy
matplotlib
pandas
scikit-learn
```

### R

```
caTools
MASS
e1071
ElemStatLearn
```

## Tech Stack

| | Python | R |
|---|---|---|
| 📄 File | `lda.py` | `lda.R` |
| 📉 LDA | `sklearn.discriminant_analysis` | `MASS::lda` |
| 🤖 Classifier | Logistic Regression | SVM (`e1071`) |
| 📊 Visualization | `matplotlib` | `ElemStatLearn` |
| 🧮 Data | `pandas` / `numpy` | base R |

## Known Issues

- **R: `ElemStatLearn` removed from CRAN** — The `ElemStatLearn` package used for visualization in `lda.R` has been archived on CRAN. Install from archive or use an alternative plotting approach.
- **R: `caTools` split** — `sample.split` may behave differently across R versions; verify your train/test ratio.

## License

MIT — see [LICENSE](LICENSE)

## Author

**Kaustabh Ganguly** ([@stabgan](https://github.com/stabgan))
