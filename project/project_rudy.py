import numpy as np
import pandas as pd
import gc
import matplotlib
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

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import validation_curve
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

#Check target variable distribution
print("\n✅ Target Variable Distribution:")
print(df['Kingdom'].value_counts())




print("\n" + "=" * 60)
print("TASK 2: DATA CLEANING")
print("=" * 60)

#Check for duplicates
print(f"Before removing duplicates: {len(df)} rows")
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

#Coerce UUU and UUC to numeric, forcing errors to NaN
df['UUU'] = pd.to_numeric(df['UUU'], errors='coerce')
df['UUC'] = pd.to_numeric(df['UUC'], errors='coerce')

#Checks # rows affected
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

#Support Vector Machine with Cross-Validation
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'class_weight': [None,'balanced'],
    'kernel': ['linear', 'rbf'] 
    
}
print("\n Running GridSearchCV for SVM - this may take a few minutes...")
#svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
#Parameters found and implemented - use if necessary!!

svm_model = SVC()
grid_svm = GridSearchCV(
    SVC(random_state=42),
    param_grid_svm,
   cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_svm.fit(X_train_scaled, y_train)
print(f"Best SVM Params: {grid_svm.best_params_}")

svm_model = grid_svm.best_estimator_

#Fit the model
svm_model.fit(X_train_scaled, y_train)

#Predict and Evaluate
y_pred_svm = svm_model.predict(X_test_scaled)

print(f"SVM Overall Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

#best SVM model predictions
cm_best = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(12, 9))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.suptitle('Tuned SVM Confusion Matrix', fontsize=16, x=0.45) 
plt.title('Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_svm)*100) + '        Macro F1 Score: {:.2f}%'.format(classification_report(y_test, y_pred_svm, target_names=le.classes_, output_dict=True)['macro avg']['f1-score']*100), fontsize=12, pad=8, color='gray')


plt.ylabel('Actual Kingdom')
plt.xlabel('Predicted Kingdom')
plt.show()
plt.close()


#Cleanup - all figures Tk roots are destroyed 

plt.close('all')
gc.collect()

if matplotlib.get_backend().lower().startswith('tk'):
    try:
        import tkinter as tk
        root = getattr(tk, '_default_root', None)
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
    except Exception:
        pass

#Classification report heatmap for SVM
#first we generated the report as a dictionary
report_dict = classification_report(y_test, y_pred_svm, target_names=le.classes_, output_dict=True)

#converted to dataframe
df_report = pd.DataFrame(report_dict).transpose()#so kingdoms are on Y-axis, metrics are on X-axis

#clean up dataframe by removing 'accuracy', 'macro avg', & 'weighted avg' to focus on kingdoms
#also exclude 'support'
df_plot = df_report.iloc[:-3, :-1] 

#plot heatmap
plt.figure(figsize=(10, 7))
# Use fixed vmin/vmax so colors are comparable across models (0-1 for precision/recall/f1)
sns.heatmap(df_plot, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0.4, vmax=1.0, cbar_kws={'label': 'Score', 'ticks':[0.0, 0.25, 0.5, 0.75, 1.0]})

#labeling
plt.title('SVM Performance Metrics by Kingdom', fontsize=15)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Kingdom', fontsize=12)
plt.show()
plt.close()


#Derive kingdom order from label encoder
kingdoms = list(le.classes_)
#imports LR F1 scores
#lr_f1 = np.load("lr_f1.npy")
# LR F1 scores - hardcoded backup)
#lr_f1 = [0.72, 0.94, 0.70, 0.74, 0.75, 0.87, 0.55, 0.52, 0.90, 0.91]

import os
lr_f1_path = os.path.join("outputs", "metrics", "lr_f1.npy")

if not os.path.exists(lr_f1_path):
    raise FileNotFoundError(
        "lr_f1.npy not found. Please run logres_rudy.py first."
    )

lr_f1 = np.load(lr_f1_path)

#Extract svm per-class f1 from classification report
svm_f1 = [report_dict[k]['f1-score'] for k in kingdoms]

x = np.arange(len(kingdoms))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, lr_f1, width, label='Logistic Regression', color='skyblue')
ax.bar(x + width/2, svm_f1, width, label='SVM (RBF)', color='navy')

ax.set_ylabel('F1-Score')
ax.set_title('Model Comparison: Logistic Regression vs. SVM')
ax.set_xticks(x)
ax.set_xticklabels(kingdoms)
ax.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.close()



#PCA - Reduce the 64 codons to 2 dimensions for visualization - excluded from report

#notice the overlap of vrt (blue) with mammals (red), primates (pink), and rodents (grey)
#Codon usage patterns show overlap, even PCA struggles to separate these rare sub-types
pca = PCA(n_components=2)
pca.fit(X_train_scaled)
X_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='tab10', alpha=0.6, s=10)

#Add a legend with actual kingdom names
handles, _ = scatter.legend_elements()
plt.legend(handles, le.classes_, title="Kingdoms", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('2D Projection of Codon Usage (PCA)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.tight_layout()
plt.show()
plt.close()






#Decision Tree with Cross-Validation
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_dt.fit(X_train_scaled, y_train)

#best model eval
best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test_scaled)

print(f"Best DT Params: {grid_dt.best_params_}")
print(f"DT Overall Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nDT Classification Report:")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

#Decision tree classification-report (heatmap)
report_dict_dt = classification_report(y_test, y_pred_dt, target_names=le.classes_, output_dict=True)
df_report_dt = pd.DataFrame(report_dict_dt).transpose()
df_plot_dt = df_report_dt.iloc[:-3, :-1]

plt.figure(figsize=(10, 7))
sns.heatmap(df_plot_dt, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0.3, vmax=1.0, cbar_kws={'label': 'Score', 'ticks':[0.0, 0.25, 0.5, 0.75, 1.0]})
plt.suptitle('Decision Tree Performance Metrics by Kingdom', fontsize=15, x=0.45) 
plt.title('Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_dt)*100) + '        Macro F1 Score: {:.2f}%'.format(classification_report(y_test, y_pred_dt, target_names=le.classes_, output_dict=True)['macro avg']['f1-score']*100), fontsize=12, pad=8, color='gray')

plt.xlabel('Metric', fontsize=12)
plt.ylabel('Kingdom', fontsize=12)
plt.show()
plt.close()


#grabs 64 codons for feature importance plot
codon_columns = [col for col in df.columns if len(col) == 3 and col.isupper()]
importances = best_dt.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10 codons

plt.figure(figsize=(10, 6))
plt.title('Top 10 Most Important Codons (Decision Tree)')
plt.barh(range(len(indices)), importances[indices], color='forestgreen', align='center')
plt.yticks(range(len(indices)), [codon_columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
plt.close()
#________________________________________________________________________________________________


#Tests for overfitting:

#Calculate accuracy for both sets
train_acc = accuracy_score(y_train, best_dt.predict(X_train_scaled))
test_acc = accuracy_score(y_test, y_pred_dt)

print(f"Decision Tree Training Accuracy: {train_acc:.4f}")
print(f"Decision Tree Testing Accuracy:  {test_acc:.4f}")
print(f"Accuracy Gap: {train_acc - test_acc:.4f}")

if train_acc > test_acc + 0.15:
    print("Conclusion: Significant OVERFITTING detected.")


#test depths from 1 to 30
param_range = np.arange(1, 31)

train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    X_train_scaled, y_train,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label="Training Score", color="blue", marker='o')
plt.plot(param_range, test_mean, label="Cross-Validation Score", color="red", marker='s')

plt.title("Validation Curve: Impact of Tree Depth on Accuracy")
plt.xlabel("Max Depth of Tree")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.show()
plt.close()
#____________________________________________________________________________________________
#Overfitting analysis:

#Dictionary to store results
comparison_data = []
models = [
    ('Decision Tree', best_dt),
    ('Tuned SVM', svm_model)
]

print("--- Overfitting Analysis ---")
for name, model in models:
    #Get predictions
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    #Calculate accuracy
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    gap = train_acc - test_acc
    
    comparison_data.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Gap': gap
    })
    
    print(f"{name}:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test:  {test_acc:.4f}")
    print(f"  Gap:   {gap:.4f}\n")

#Convert to dataframe
df_comp = pd.DataFrame(comparison_data)


#Compare train vs test accuracy for dt & svm
labels = df_comp['Model']
train_scores = df_comp['Train Accuracy']
test_scores = df_comp['Test Accuracy']

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, train_scores, width, label='Train (Memorization)', color='#66b3ff')
rects2 = ax.bar(x + width/2, test_scores, width, label='Test (Generalization)', color='#004c99')

ax.set_ylabel('Accuracy')
ax.set_title('Determining Model Robustness: DT vs. SVM')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add a horizontal line at 1.0 to indicate perfect accuracy
ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
plt.show()
plt.close()