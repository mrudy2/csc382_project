import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import learning_curve

import warnings
from sklearn.exceptions import ConvergenceWarning
import os


DATA_PATH = os.path.join("data", "codon_usage.csv")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("\n✅ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Column names: {list(df.columns)}")

#Check first few rows
print("\n✅ First 5 rows:")
print(df.head())

#Summary statistics
print("\n✅ Summary Statistics:")
print(df.describe())

#Missing values
print("\n✅ Missing Values Check:")
print(df.isnull().sum())
print(f"Total missing values: {df.isnull().sum().sum()}")

#Check target var distribution
print("\n✅ Target Variable Distribution:")
print(df['Kingdom'].value_counts())




print("\n" + "=" * 60)
print("TASK 2: DATA CLEANING")
print("=" * 60)

#Checks for duplicates
print(f"Before removing duplicates: {len(df)} rows")
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

#Coerce UUU and UUC to numeric, forcing errors to NaN
df['UUU'] = pd.to_numeric(df['UUU'], errors='coerce')
df['UUC'] = pd.to_numeric(df['UUC'], errors='coerce')

#Checks # of rows affected
print(f"\nRows with errors in 'UUU' or 'UUC': {df[['UUU', 'UUC']].isnull().any(axis=1).sum()}")

#Drops rows with NaN values
df.dropna(subset=['UUU', 'UUC'], inplace=True)
print(f"After removing rows with errors: {len(df)} rows")

#remove plasmids - may cause imbalance
df = df[df['Kingdom'] != 'plm'] 
print(f"After removing plasmids: {len(df)} rows")

#Limit DNAtypes to Nuclear, Mitochondrial, Chloroplast (0,1,2)
valid_dna_types = [0,1,2]
df = df[df['DNAtype'].isin(valid_dna_types)]
print(f"After limiting DNA types: {len(df)} rows")


print("\n✅ After preprocessing - Target Variable Distribution:")
print(df['Kingdom'].value_counts())


#Data leakage - strong correlation of DNAtype with Kingdom - remove!!! (no need to one-hot encode)
#___________________________________________________________________________________________

#Ensure only codons columns are seleted
codon_columns = [col for col in df.columns if len(col) == 3 and col.isupper()]
X = df[codon_columns]
X = X.dropna()

#Isolate features (X) and Target (y)
y = df['Kingdom']
print("\n✅ Feature Selection After Cleaning:")
print(f"Features (X) shape: {X.shape}")
print(f"Features names: {list(X.columns)}")
print(f"Target (y) shape: {y.shape}")

#Encode the Kingdom labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print("\n✅ Split Data into Train/Test Sets:")

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n✅ Z-Score Standardization Applied:")
#_____________________________________________________________________________________________

# Logistic Regression
print("\n=====Starting Logistic Regression=====")
lr = LogisticRegression(C=1.0, class_weight=None, solver='lbfgs', random_state=42, max_iter=1000)
#hardcoded for efficiency but match GridSearchCV results - see terminal output to confirm
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))



#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.ylabel('Actual Kingdom')
plt.xlabel('Predicted Kingdom')
plt.suptitle('Logistic Regression Confusion Matrix', fontsize=16, x=0.45) 
plt.title('Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100) + '        Macro F1 Score: {:.2f}%'.format(classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)['macro avg']['f1-score']*100), fontsize=12, pad=8, color='gray')
plt.show()

#To visualize the classification report, converted to DataFrame & plot heatmap
#first convert  report to dictionary
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

#Derive kingdom order from label encoder, for lr_f1 reproducibility add-on
kingdoms = list(le.classes_)

#Extract per-class F1 scores & save for downstream comparison
lr_f1 = [report_dict[k]['f1-score'] for k in list(le.classes_)]
np.save("lr_f1.npy", lr_f1)

#Remove 'accuracy' and 'macro/weighted avg' to focus on kingdoms
df_report = pd.DataFrame(report_dict).iloc[:-1, :-3].T 

plt.figure(figsize=(10, 6))
sns.heatmap(df_report, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0.4, vmax=1.0, cbar_kws={'label': 'Score', 'ticks':[0.0, 0.25, 0.5, 0.75, 1.0]})
plt.title('Logistic Regression Performance Metrics by Kingdom')
plt.show()



#Suppress warnings for a clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#______________________________________________________________________________________________


#Hyperparameter Tuning with GridSearchCV

#Simplified grid - avoids the deprecated 'penalty' parameter
param_grid_lr = {
    'C': [0.1, 1, 10],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42),
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best Params: {grid_search.best_params_}")
print(f"Best Cross-Validation Score (F1 Macro): {grid_search.best_score_:.4f}")


#Learning curve plotted
train_sizes, train_scores, test_scores = learning_curve(
    grid_search.best_estimator_, X_train_scaled, y_train, cv=5)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
plt.title('Learning Curve: Is more data needed?')
plt.xlabel('Training Samples')
plt.ylabel('Score')
plt.legend()
plt.show()

print("\n✅ Logistic Regression Done")

# --- Overfitting diagnostics ---
best_lr = grid_search.best_estimator_

#Training vs Test accuracy
train_pred = best_lr.predict(X_train_scaled)
test_pred = best_lr.predict(X_test_scaled)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
cv_scores = cross_val_score(best_lr, X_train_scaled, y_train, cv=5, scoring='f1_macro')

print(f"\n=== Overfitting Diagnostics ===")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Mean CV F1 (train folds): {cv_scores.mean():.4f}")

gap = train_acc - test_acc
if gap > 0.05:
    print("Warning: Training accuracy is substantially higher than test accuracy — possible overfitting.")
else:
    print("No strong overfitting detected")


#Reproducibility Add-ons:
import os
import numpy as np

os.makedirs("outputs/metrics", exist_ok=True)

lr_f1 = [report_dict[k]['f1-score'] for k in kingdoms]

np.save(os.path.join("outputs", "metrics", "lr_f1.npy"), lr_f1)

print("\n✅ Saved Logistic Regression F1 scores to outputs/metrics/lr_f1.npy")
print("\n logres_rudy.py completed successfully")