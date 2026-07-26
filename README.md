# Bank Credit Risk Prediction - Complete ML Pipeline

## 📖 Project Overview
This project contains a comprehensive machine learning pipeline designed to predict bank credit risk[cite: 2]. The primary objective is to build a predictive model that identifies customers with a high likelihood of credit default, enabling proactive risk mitigation[cite: 2]. 

## 🎯 Business Problem & Metrics
Financial institutions face the ongoing challenge of predicting credit defaults[cite: 2]. Early identification helps minimize financial losses, but the model must strike a balance between false positives (incorrectly rejecting good customers, leading to business loss) and false negatives (missing bad customers, leading to financial loss)[cite: 2].

**Success Metrics Evaluated:**
*   **AUC-ROC:** The primary metric for overall discriminative ability[cite: 2].
*   **Precision:** Of the predicted defaults, how many were actual defaults?[cite: 2]
*   **Recall:** Of the actual defaults, how many did the model successfully catch?[cite: 2]

## 📂 Data Sources
The model integrates data from three separate sources, which are merged via a common `customer_no` key[cite: 2]:
1.  **`Cust_Demographics.csv`**: Contains customer demographic information alongside the target variable (`Bad_label`)[cite: 2].
2.  **`Cust_Account.csv`**: Includes account details, historical payment data, and balances[cite: 2].
3.  **`Cust_Enquiry.csv`**: Contains historical credit enquiry information[cite: 2].

## 🛠️ Machine Learning Pipeline
The notebook outlines a structured approach to process the data and train the predictive models[cite: 2]:

### 1. Data Integration & Cleaning
*   Data from account and enquiry tables were aggregated to create customer-level summary statistics (e.g., total/average high credit, total past due, and enquiry counts) before merging[cite: 2].
*   **Robust NaN Handling:** Proper missing value imputation was a critical step for model success[cite: 2]. Rows missing the target variable (`Bad_label`) were dropped[cite: 2]. Remaining numeric missing values were imputed using the median/mean, and categorical missing values were filled with the label 'Unknown'[cite: 2].

### 2. Feature Engineering & Preprocessing
*   The dataset was split into a 70% training set and a 30% testing set using a stratified split[cite: 2].
*   Features were standardized using `StandardScaler` to ensure uniform scale across the dataset[cite: 2].

### 3. Predictive Modeling
Two algorithms were trained and compared. Both models utilized `class_weight='balanced'` to account for the heavy class imbalance in the target variable (967 non-defaults vs. 33 defaults)[cite: 2]:
*   **Logistic Regression:** Serves as the interpretable baseline model outputting default probabilities[cite: 2].
*   **Random Forest:** An ensemble classifier utilized to capture non-linear relationships and extract feature importance[cite: 2].

## 📊 Results & Evaluation
The models were evaluated primarily on their AUC-ROC scores to account for the imbalanced nature of the data[cite: 2]:

| Model | Train Accuracy | Test Accuracy | AUC-ROC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7714 | 0.7700 | 0.5341[cite: 2] |
| **Random Forest** | 1.0000 | 0.9667 | 0.6334[cite: 2] |

*   **Best Model:** The **Random Forest** algorithm proved to be the best model, achieving the highest AUC-ROC score of 0.6334 and an accuracy of 96.67% on the test set[cite: 2].
*   **Key Drivers of Risk:** Feature importance extraction revealed that `feature_7`, `customer_no`, `feature_30`, `feature_52`, and `feature_44` are the top indicators of credit risk within this dataset[cite: 2].

## 💻 Dependencies
To run this notebook, the following Python libraries are required:
*   `pandas`[cite: 2]
*   `numpy`[cite: 2]
*   `matplotlib` & `seaborn` (for exploratory data analysis and visual reports)[cite: 2]
*   `scikit-learn` (for data splitting, imputation, scaling, modeling, and evaluation metrics)[cite: 2]
