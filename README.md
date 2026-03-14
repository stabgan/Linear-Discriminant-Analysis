# 📊 Linear Discriminant Analysis (LDA)

Dimensionality reduction and multi-class classification using **Linear Discriminant Analysis**, implemented in both Python and R. Uses the UCI Wine dataset to reduce 13 features down to 2 linear discriminants, then classifies wine segments with decision-boundary visualizations.

## How It Works

1. Load the Wine dataset (13 chemical features, 3 customer segments)
2. Split into 80/20 train/test and apply feature scaling
3. Apply LDA to project data onto 2 discriminant components
4. Train a classifier (Logistic Regression in Python, SVM in R)
5. Visualize decision boundaries for both train and test sets

## Tech Stack

| | Python | R |
|---|---|---|
| **File** | `lda.py` | `lda.R` |
| **LDA** | `sklearn.discriminant_analysis` | `MASS::lda` |
| **Classifier** | Logistic Regression | SVM (`e1071`) |
| **Visualization** | matplotlib | ElemStatLearn |

### Python Dependencies

```
numpy
matplotlib
pandas
scikit-learn
```

### R Dependencies

```r
caTools
MASS
e1071
ElemStatLearn
```

## Quick Start

### Python

```bash
pip install numpy matplotlib pandas scikit-learn
python lda.py
```

### R

```r
# Install packages if needed
install.packages(c("caTools", "e1071", "ElemStatLearn"))

source("lda.R")
```

## Dataset

**Wine.csv** — 178 samples, 13 features (Alcohol, Malic Acid, Ash, Magnesium, Phenols, etc.), 3 target classes (`Customer_Segment`). Based on the [UCI Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine).

## License

MIT — see [LICENSE](LICENSE)

## Author

**Kaustabh Ganguly** ([@stabgan](https://github.com/stabgan))
