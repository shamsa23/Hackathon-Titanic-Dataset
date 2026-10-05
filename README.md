# Titanic Survival Prediction 🚢

*Built at the International Women's Day Hackathon 2026 (QMUL) · Theme: Mathematics & Computing for Real-World Problems*

## Motivation ⚙️

### Chosen data-related problem

> **"Which features most strongly predict whether a passenger survived the Titanic, and how accurately can machine learning models predict it?"**

The sinking of the Titanic is one of the most studied datasets in machine learning. Survival was not random: who you were (gender, age, ticket class) shaped your chances. This project uses supervised classification to find which passenger features matter most, and uses explainability (SHAP) to show why the best model makes its predictions.

### Chosen dataset

The dataset used was the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) `train.csv` (891 passengers). The target variable is `Survived` (0 = did not survive, 1 = survived). Features include passenger class, sex, age, fare, port of embarkation, and family aboard.

### Chosen machine learning algorithms

| Model | Why I chose it |
|---|---|
| **Logistic Regression** | Simple, interpretable baseline for binary classification |
| **Random Forest** | Captures non-linear patterns and interactions, and is robust to overfitting |
| **XGBoost** | Gradient boosting, often the strongest model on tabular data |

Models were trained on an 80:20 train-test split (712 train / 179 test, `random_state=42`) and compared using accuracy, ROC-AUC and confusion matrices.

## Project Objectives 🎯

1. Explore the data (EDA): survival rates by gender, class and age group.
2. Engineer new features: passenger title and family size.
3. Train and compare three classifiers.
4. Evaluate with accuracy, ROC-AUC and confusion matrices.
5. Use SHAP to explain which features drive the best model's predictions.

## Environment 👩🏻‍💻

<p align="center">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/>
</p>

## Stack 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=flat" alt="matplotlib"/>
  <img src="https://img.shields.io/badge/seaborn-4C72B0?style=flat" alt="seaborn"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/SHAP-FF0051?style=flat" alt="SHAP"/>
</p>

## Repository Structure 🌲

```
.
├── .gitattributes
├── Option_5.ipynb
├── README.md
├── test.txt
└── test2.txt
```

## Method 🧪

| Stage | What I did |
|---|---|
| **EDA** | Survival rates by gender, class and age group; missing-value heatmap |
| **Cleaning** | Filled `Age` with the median, `Embarked` with the mode, dropped `Cabin` (mostly missing) |
| **Feature engineering** | Extracted Title from names (Mr, Mrs, Miss, Rare), created FamilySize, binned AgeGroup |
| **Encoding** | One-hot encoded Sex, Embarked, Title, AgeGroup and Pclass |
| **Modelling** | Logistic Regression, Random Forest, XGBoost |
| **Explainability** | SHAP summary plots for the Random Forest |

## Results 📊

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| **Logistic Regression** | 0.7933 | 0.8799 |
| **Random Forest** | 0.8380 | 0.9055 |
| **XGBoost** | 0.8268 | 0.8979 |

- **Random Forest performed best** on both accuracy and ROC-AUC.
- XGBoost was a close second, and Logistic Regression was the weakest but still a solid baseline.
- Tree-based models beating the linear model suggests non-linear relationships between features and survival.

<!-- Optional: add your plots. Save them into an images/ folder, then uncomment:
![Confusion matrices](images/confusion_matrices.png)
![SHAP summary](images/shap_summary.png)
-->

## Main Findings 🔍

- **Overall survival rate was ~38%**, so the classes are imbalanced.
- **Gender was the strongest predictor**: `Sex_male` had the largest impact in SHAP, with being male lowering survival probability.
- **Fare and passenger class** mattered: higher fares and 1st class increased survival, and 3rd class decreased it.
- **Age** helped most for children, supporting a "women and children first" pattern.
- **Title** (Mr / Mrs / Miss) reinforced the gender and age effects.
- **FamilySize** had a smaller, more mixed influence.

## Recommendations for Improvements 📈

- **Hyperparameter tuning** (e.g. `GridSearchCV`) for Random Forest and XGBoost.
- **Cross-validation** for a more reliable estimate than a single train-test split.
- **More feature engineering**, e.g. cabin deck, ticket groups, and keeping `SibSp` / `Parch` separately.
- **Better missing-value handling**, e.g. predicting age from title and class instead of using the overall median.

## Reflection 🪞

This hackathon project gave me hands-on experience with the full machine learning workflow, from exploring and cleaning data to engineering features, comparing models and explaining the results. Using SHAP showed me that a high-accuracy model is far more useful when you can see *why* it predicts what it does. Next, I want to explore hyperparameter tuning, cross-validation and further explainability techniques.
