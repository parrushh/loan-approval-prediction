# 🏦 CrediDecision: Loan Approval Prediction System

An end-to-end Machine Learning classification pipeline built in Python to predict loan approval outcomes based on applicant financial profiles, demographics, and credit indicators.

---

## 📊 Project Overview
Financial institutions evaluate numerous applicant parameters to assess creditworthiness and manage lending risk. This project automates the qualification workflow by analyzing historical applicant data, handling missing information, engineering non-linear feature interactions, and evaluating supervised classification algorithms to optimize loan eligibility decisions.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning & Pipelines:** Scikit-Learn
  * Data Imputation: `SimpleImputer`
  * Feature Scaling & Encoding: `StandardScaler`, `OneHotEncoder`, `LabelEncoder`
  * Classifiers: `LogisticRegression`, `GaussianNB`, `KNeighborsClassifier`
  * Optimization: `GridSearchCV`, `Pipeline`

---

## ⚙️ Workflow & Methodology

1. **Exploratory Data Analysis (EDA):**
   * Visualized target class distribution (`Loan_Approved`) and key demographics (`Gender`, `Employer_Category`, `Loan_Purpose`).
   * Conducted bivariate analysis using grouped box plots and correlation heatmaps to assess feature interactions with loan approvals.

2. **Data Cleaning & Feature Engineering:**
   * Handled missing numerical and categorical values using mean and mode imputation via `SimpleImputer`.
   * Dropped non-informative identifier fields (`Applicant_ID`).
   * Applied One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) and Label Encoding (`LabelEncoder`) to prepare features for modeling.
   * Engineered squared interaction terms (`DTI_Ratio_sq`, `Credit_Score_sq`) to capture non-linear behavioral relationships.

3. **Model Architecture & Evaluation:**
   * Partitioned dataset into 80/20 train-test splits (`train_test_split`).
   * Standardized features using `StandardScaler` to prevent feature scale bias.
   * Benchmarked baseline performance using **Logistic Regression** and **Gaussian Naive Bayes**.
   * Constructed an automated `Pipeline` paired with `GridSearchCV` (5-fold cross-validation) to optimize the $k$-nearest neighbors hyperparameter (`n_neighbors`).
   * Evaluated model outputs using Accuracy, Precision, Recall, and F1-Score metrics.

---

## 🚀 How to Run locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/CrediDecision.git](https://github.com/YOUR_USERNAME/CrediDecision.git)
   cd CrediDecision