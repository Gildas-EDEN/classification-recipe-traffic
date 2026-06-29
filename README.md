# Predicting High Site Traffic — Recipe Classification

A supervised classification project that predicts whether a recipe will generate high traffic on a cooking website, based on its nutritional features and dish category. The business goal: help surface the recipes most likely to attract visitors.

## Problem statement

Given a recipe's information (calories, macronutrients, number of servings, dish category), can we predict the target variable `high_traffic` (high traffic or not)? The challenge is twofold: predict accurately, but also **understand which variables drive traffic**.

## Data

- `recipe_site_traffic` dataset: **947 recipes**, provided in [`data/recipe_site_traffic.csv`](data/recipe_site_traffic.csv).
- Variables: `calories`, `carbohydrate`, `sugar`, `protein`, `servings`, `category`, and the target `high_traffic`.
- After handling missing values, **895 rows kept (94.5% of the initial dataset)**.

## Approach

1. **Data wrangling**: handling missing values (imputing `high_traffic` as `Low`, dropping the rest), removing duplicates.
2. **Exploratory Data Analysis (EDA)**: descriptive statistics, visualizations by category and target distribution.
3. **Preparation**: binary recoding of the target, one-hot encoding of categories, numeric extraction of `servings`, 80/20 train/test split.
4. **Modeling**:
   - **Linear Probability Model (LPM)** and **Logit model** (statsmodels), with **marginal effects** and significance thresholds.
   - Comparison of the marginal effects of both models.
   - **Random Forest** (scikit-learn) and feature importance analysis.
5. **Interpretation**: identifying the most influential variables (calories, carbohydrate, and sugar stand out as the most decisive).

## Tech stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `statsmodels`

## Repository structure

```
.
├── recipe_traffic_classification.ipynb   # Full notebook (analysis + models + outputs)
├── data/
│   └── recipe_site_traffic.csv           # Dataset
├── README.md
└── .gitignore
```

## Reproduce the analysis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
jupyter notebook recipe_traffic_classification.ipynb
```

The notebook reads `data/recipe_site_traffic.csv` directly, so no additional setup is required.

## Author

**Gildas Edenakpo** — M2 Econometrics & Data Science, Aix-Marseille School of Economics (AMSE).
