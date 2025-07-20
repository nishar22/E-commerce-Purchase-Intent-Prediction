#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

#load the dataset
df = pd.read_csv("online_shoppers_intention.csv")
df.head()

##Data Preprocessing & EDA
# Check data types, missing values, and structure
df.info()

# View basic statistics of numerical columns
df.describe()

# Check if there are any missing values
df.isnull().sum()

#Target variable distribution
sns.countplot(x='Revenue', data=df)
plt.title("Class Balance (Revenue)")

# Convert categorical and boolean variables using one-hot encoding
categorical = df.select_dtypes(include=['object', 'bool']).columns
df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)

# Convert categorical and boolean variables using one-hot encoding
X = df_encoded.drop('Revenue_True', axis=1)
y = df_encoded['Revenue_True']

# Scale the numerical features to bring them to a similar range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Since purchase events are less frequent, balance the dataset using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Check class distribution after balancing
print("After SMOTE:")
print("Class 0:", sum(y_resampled == 0))
print("Class 1:", sum(y_resampled == 1))

#Train-Test Split
# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

##train models
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)

##Evaluate Models with ROC and Metrics
# Define a function to evaluate any model
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]   # Probability estimates
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')

# Evaluate each model
evaluate_model(lr, "Logistic Regression")
evaluate_model(rf, "Random Forest")
evaluate_model(svm, "SVM")

# Final ROC formatting
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

