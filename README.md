# SQUAD x insightfactory.ai Datathon — 1st place

Repository for the winning entry of the SQUAD x insightfactory.ai Datathon (May 2025).

**Languages / Libraries:** Python, XGBoost, scikit-learn, seaborn

**Brief:** A concise project that explores an anonymised dataset with 19 features (`X1`..`X19`) and a binary target `Y`. The goal was to analyse the data, handle class imbalance and missing values, and produce high-quality predictions evaluated by F1 score.

**Winner note:** This submission placed 1st in the competition.

**Repository files**
- `train.csv` — training data (features `X1`..`X19`, target `Y`)
- `test.csv` — test data (features only)
- `sample_submission.csv` — sample submission format
- `main.ipynb` — analysis notebook (EDA, preprocessing, modeling, predictions)
- `prediction.csv` — example output produced by the notebook

**Dataset description**
The dataset contains 19 anonymous numerical features (`X1`..`X19`) and a binary target `Y`. No feature metadata was provided; the task focused on inferential EDA and robust modeling while avoiding data leakage.

**Evaluation metric**: F1 score (harmonic mean of precision and recall).

**High-level workflow**
1. Exploratory data analysis (EDA) to understand distributions, missingness, correlations, and potential feature patterns.
2. Data cleaning and preprocessing: missing value handling, outlier inspection, and scaling where appropriate.
3. Model development: XGBoost classifier with class weighting to address target imbalance.
4. Model validation and selection using cross-validation / hold-out validation to avoid leakage.
5. Produce predictions for `test.csv` and save as `prediction.csv` in the required `ID,Y` format.

EDA & Preprocessing highlights
- **Missing values:** Identified and imputed (feature-wise strategies) to preserve signal while avoiding introducing bias.
- **Feature patterns:** Plotted distributions and pairwise relationships to detect skew and correlation structure; transformed skewed variables where helpful.
- **Scaling:** Applied scaling for features used by models or pipelines that are sensitive to scale.
- **Leakage:** Carefully avoided leakage by performing preprocessing inside cross-validation folds (when tuning) and not using target-derived information.

Modeling summary
- **Model:** XGBoost classifier (gradient-boosted trees).
- **Imbalance handling:** Class weighting to penalise misclassification of the minority class.
- **Validation:** Used cross-validation / hold-out to evaluate performance. Final reported validation F1: **0.44**.
- **Feature engineering & tuning:** Simple, interpretable feature checks and modest hyperparameter tuning to avoid overfitting given the anonymised features.

Reproducing the results
1. Create and activate a Python environment (PowerShell example):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (either create a `requirements.txt` with the packages below or install directly):

```
pip install numpy pandas scikit-learn xgboost seaborn matplotlib jupyter
```

3. Open and run the notebook `main.ipynb` (recommended) or run the provided scripts if available.

4. The notebook performs EDA, preprocessing, model training, and writes predictions to `prediction.csv`.

Submission format
- The submission file must contain a header and rows ordered by `ID` (ascending):

```
ID,Y
2,0
5,0
6,0
...
```

Tips and reproducibility notes
- Ensure all preprocessing steps that use information from `Y` (e.g., scaling based on class-specific statistics) are fitted inside cross-validation folds to prevent leakage.
- Ordering: sort by `ID` ascending before writing the final CSV.
- Random seeds: set seeds for deterministic model runs where possible (XGBoost `random_state`, NumPy, scikit-learn) to improve reproducibility.

Potential next improvements
- More extensive hyperparameter search (Bayesian / grid search) with nested cross-validation.
- Advanced imbalance handling: SMOTE, ensemble balancing, or cost-sensitive learning.
- Alternative models and stacking ensembling for incremental gains.

Acknowledgements
Thanks to the SQUAD and insightfactory.ai datathon organisers and to my teammates for the collaboration.

Contact
For questions or reproducibility help, open an issue in this repository or contact the project owner.
