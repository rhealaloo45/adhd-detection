# ADHD detection with classical ML

This project compares three classifiers (Gaussian Naive Bayes, an SVM and a Random Forest) at predicting ADHD diagnosis from phenotypic data in the ADHD-200 dataset: age, gender, handedness, ADHD rating scores, IQ scores and medication status. Random Forest came out best, at 87.3% accuracy on a held-out test set.

It was a four-person minor project in the third year (semester VI) of Information Technology engineering at K. J. Somaiya Institute of Technology, 2023–24. The synopsis is in [`SEM6_PBL_Final Synonpsis.pdf`](SEM6_PBL_Final%20Synonpsis.pdf).

## Results

All three notebooks hold out 20% of the 550 rows for testing (`train_test_split(test_size=0.2, random_state=42)`), so each test set has 110 rows. The numbers below are copied from the saved notebook outputs.

| Model | Notebook | Test accuracy | Training accuracy |
|---|---|---|---|
| **Random Forest** | [`ADHD_RF.ipynb`](ADHD_RF.ipynb) | **87.3%** | 85.5% (5-fold CV mean) |
| SVM (RBF kernel, scaled features) | [`ADHD_SVM_NEW.ipynb`](ADHD_SVM_NEW.ipynb) | 80.0% | 74.3% |
| Gaussian Naive Bayes | [`ADHD_NB.ipynb`](ADHD_NB.ipynb) | 60.0% | 57.7% |

Per-class results for the Random Forest on the 110-row test set:

| DX class | Precision | Recall | F1 | Test rows |
|---|---|---|---|---|
| 0 | 0.96 | 0.96 | 0.96 | 73 |
| 1 | 0.81 | 0.81 | 0.81 | 26 |
| 3 | 0.45 | 0.45 | 0.45 | 11 |

The classes are imbalanced, and the smallest class is where every model struggles. Each notebook cleans the data slightly differently (the SVM notebook ends up with 549 rows), so the three test splits are not exactly the same rows.

## How it works

1. **Data.** Each notebook loads `cleaned_data_DX.csv`, a cleaned copy of the ADHD-200 phenotypic table (550 rows). The target is `DX`. Features: `Gender`, `Age`, `Handedness`, `ADHD Measure`, `ADHD Index`, `Inattentive`, `Hyper/Impulsive`, `IQ Measure`, `Verbal IQ`, `Performance IQ`, `Full4 IQ`, `Med Status`.
2. **Cleaning.** Depending on the notebook, missing values are either filled with the column's mode or dropped. The Random Forest notebook also replaces the dataset's `-999` placeholders, rounds `Handedness` to its categories, and one-hot encodes `Gender` and `Handedness`, which gives 15 features.
3. **Models.** scikit-learn `RandomForestClassifier(random_state=42)`, `SVC()` on `StandardScaler` features, and `GaussianNB()`, each scored with accuracy and a classification report.
4. **Spot checks.** The Random Forest notebook ends with hand-entered patient profiles. Its markdown records whether each prediction matched the known label.

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib and Seaborn, run in Jupyter and Google Colab.

## Run it

The notebooks were written for Google Colab and read the data from `/content/cleaned_data_DX.csv`. That CSV is not in this repository.

1. Get the ADHD-200 phenotypic data and clean it into `cleaned_data_DX.csv` with the columns listed above.
2. Open a notebook in Colab (or Jupyter), upload the CSV, and change the `read_csv` path if you are not running in Colab.
3. Run all cells.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook ADHD_RF.ipynb
```

## A note on the labels

The notebooks print predictions as severity levels ("No adhd", "Mild", "Moderate", "Severe"). In the ADHD-200 data dictionary, `DX` is a diagnosis category: 0 is typically developing, 1 is ADHD-Combined, 2 is ADHD-Hyperactive/Impulsive and 3 is ADHD-Inattentive. Read the predicted classes as diagnosis categories, not severity levels.
