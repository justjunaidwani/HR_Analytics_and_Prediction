# IBM HR Analytics: Employee Attrition & Performance

## Project Overview

This project focuses on analyzing and predicting employee attrition using the **IBM HR Analytics Employee Attrition & Performance** dataset. The goal is to identify key factors that lead to employee turnover and build a machine learning model capable of accurately predicting which employees are likely to leave the company (attrite). This insight allows HR departments to proactively intervene and implement targeted retention strategies.

## Table of Contents

1.  [Data Source](#data-source)
2.  [Objectives](#objectives)
3.  [Methodology](#methodology)
4.  [Key Findings (Exploratory Data Analysis)](#key-findings-exploratory-data-analysis)
5.  [Modeling & Results](#modeling-results)
6.  [Setup and Installation](#setup-and-installation)
7.  [Project Structure](#project-structure)

## Data Source

The analysis is based on the `HR Employee Attrition.csv` dataset, which was provided by the Unified Mentor where I worked as an Intern. This dataset contains 1,470 employee records and 35 features, including demographic, job role, salary, and satisfaction metrics.

## Objectives

* **Exploratory Data Analysis (EDA):** Identify significant correlations and trends between employee features and attrition.
* **Feature Engineering:** Transform and create features to improve model performance (e.g., log-transformation for skewed data, age binning).
* **Predictive Modeling:** Develop a robust classification model (starting with Logistic Regression) to predict the binary target variable `Attrition` (Yes/No).
* **Performance Evaluation:** Focus on model metrics such as **Recall** for the minority (Attrited) class, as predicting true positives is critical for a high-cost problem like employee turnover.

## Methodology

The project follows a standard data science pipeline:

1.  **Data Loading and Cleaning:**
    * Loaded the `HR Employee Attrition.csv` dataset.
    * Droppped non-informative columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).
    * Converted the target variable `Attrition` from categorical (`Yes`/`No`) to numerical (`1`/`0`).

2.  **Exploratory Data Analysis (EDA):**
    * Checked for class imbalance (approx. 84% 'No' Attrition, 16% 'Yes' Attrition).
    * Analyzed attrition rates across various categorical and numerical features.
    * Addressed skewness in features like `MonthlyIncome` and `NumCompaniesWorked` using log-transformation (`np.log1p`).

3.  **Preprocessing and Feature Engineering:**
    * Created `Age_Bins` feature to group employees into age categories.
    * Applied **One-Hot Encoding** to all categorical features.
    * Applied **Standard Scaling** to all numerical features to ensure all features contribute equally to the model.

4.  **Model Training:**
    * Split the data into training and testing sets.
    * Trained a **Logistic Regression** model as a baseline classifier.

## Key Findings (Exploratory Data Analysis)

Based on the initial analysis, several factors showed a strong relationship with attrition:

* **Age:** The **29-38** age group exhibited the highest attrition rate (42.62%), followed closely by the **18-28** group (30.80%). 
* **Department:** The **Research & Development** department had the lowest attrition rate (around 14%), despite having the largest workforce. 
* **Job Satisfaction:** A significant number of attrited employees reported **Low Job Satisfaction**, highlighting satisfaction metrics as crucial retention indicators. 

## Modeling & Results

The initial baseline model was a Logistic Regression classifier.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Testing Accuracy** | ~88.5% | High overall accuracy due to class imbalance. |
| **Recall (Class 1: Attrition)** | **0.44** | Indicates that the model only correctly identified 44% of the actual attrited employees. |

### Model Optimization Note

Due to the severe class imbalance, the low **Recall** score for the positive class (Attrition) is the primary area for improvement. Future model iterations should focus on:

1.  Employing techniques to handle imbalance, such as **SMOTE (Synthetic Minority Over-sampling Technique)**.
2.  Utilizing the `class_weight='balanced'` parameter in the classifier to penalize misclassifications of the minority class more heavily.
3.  Experimenting with more complex models like Random Forests or Gradient Boosting.

## Setup and Installation

To run the Jupyter notebook (`IBM HR Anlaytics.ipynb`), you will need the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

├── IBM HR Anlaytics.ipynb  # Main analysis and modeling notebook

├── HR Employee Attrition.csv # The dataset used for the project

└── README.md













