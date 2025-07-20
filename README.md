# E-commerce-Purchase-Intent-Prediction


This project aims to predict whether a visitor to an e-commerce website will make a purchase during their session based on behavioral data. It demonstrates a full machine learning pipeline from data exploration and preprocessing to modeling and evaluation.



## Objective

To build a binary classification model that predicts **purchase intent (`Revenue`)** of a website visitor using session-based features such as:
- Pages visited 
- Time spent on different page types
- Bounce and exit rates
- Traffic source, browser, month, weekend flag, and more



##  Dataset

- **Source**:: [Kaggle – Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/adilshamim8/online)
- **Author**: [adilshamim8 on Kaggle](https://www.kaggle.com/adilshamim8) 
- **Format**: CSV (12,330 rows × 18+ features)
- **Target Variable**: `Revenue` (True = Purchase, False = No Purchase)

---

## Methodology

### 1.  Exploratory Data Analysis (EDA)
- Checked for missing values and data types
- Visualized class imbalance (only ~15% purchases)

### 2.  Preprocessing
- One-hot encoded all categorical and boolean variables
- Scaled numeric features using `StandardScaler`
- Balanced the dataset using **SMOTE**

### 3.  Model Building
Three classification models were trained and compared:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

### 4.  Model Evaluation
Each model was evaluated using:
- Accuracy
- Confusion Matrix
- ROC-AUC Score
- Precision, Recall, F1-score



##  Results

| Model               | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| 0.84     | 0.92    |
| Random Forest       | 0.93     | 0.99    |
| SVM                 | 0.88     | 0.94    |




