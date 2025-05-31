#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc


# ## Load Dataset

# In[3]:


df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# ## Preview Data

# In[4]:


df.head()              # Lihat 5 baris pertama
df.info()              # Cek struktur data
df.describe()          # Statistik deskriptif kolom numerik
df.isnull().sum()      # Cek missing values


# ### Cek Tipe Data

# In[5]:


df.dtypes


# ### Konversi TotalCharges ke Numerik

# In[6]:


# Mengubah TotalCharges ke float (karena awalnya object), dan tangani error non-numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Cek kembali missing value setelah konversi
df.isnull().sum()


# ### Tangani Missing Value

# In[7]:


df.dropna(inplace=True)


# In[8]:


df.isnull().sum()


# ### Cek Statistik Deskriptif

# In[9]:


df.describe()


# ### Visualisasi Distribusi Numerik (Univariate Analysis)

# Histogram

# In[10]:


numerik_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

df[numerik_cols].hist(bins=30, figsize=(12, 6), color='skyblue')
plt.suptitle("Distribusi Fitur Numerik")
plt.tight_layout()
plt.show()


# Boxplot untuk Deteksi Outlier

# In[11]:


plt.figure(figsize=(12, 4))
for i, col in enumerate(numerik_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col], color='lightcoral')
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()


# ### Cek Korelasi antar Fitur Numerik

# In[12]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[numerik_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# ### Analisis Fitur Kategorikal (Univariate dan Bivariate)

# Cek Jumlah Kategori

# In[13]:


kategori_cols = df.select_dtypes(include='object').columns.tolist()
kategori_cols.remove('customerID')  # kita abaikan karena bukan fitur analisis

for col in kategori_cols:
    print(f"\n{col}:\n", df[col].value_counts())


# Visualisasi Distribusi Churn berdasarkan Tipe Kontrak

# In[14]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Distribusi Churn berdasarkan Tipe Kontrak')
plt.xlabel('Tipe Kontrak')
plt.ylabel('Jumlah Pelanggan')
plt.legend(title='Churn')
plt.tight_layout()
plt.show()


# Visualisasi Univariate Fitur Kategorikal

# In[15]:


plt.figure(figsize=(15, 20))
for i, col in enumerate(kategori_cols):
    plt.subplot(6, 3, i + 1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
    plt.xticks(rotation=45)
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()


# In[16]:


kategori_cols = df.select_dtypes(include='object').columns.tolist()
kategori_cols.remove('customerID')  # customerID bukan fitur analisis

for col in kategori_cols:
    print(f"Distribusi nilai '{col}':")
    print(df[col].value_counts())
    print("\n" + "-"*40 + "\n")


# Visualisasi Bivariate (Churn vs Fitur Kategorikal)

# In[17]:


plt.figure(figsize=(15, 20))
for i, col in enumerate(kategori_cols):
    plt.subplot(6, 3, i + 1)
    sns.countplot(data=df, x=col, hue='Churn', palette='Set1')
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Churn')
plt.tight_layout()
plt.show()


# In[18]:


for col in kategori_cols:
    print(f"Distribusi Churn berdasarkan '{col}':")
    print(pd.crosstab(df[col], df['Churn'], normalize='index').round(2))
    print("\n" + "-"*50 + "\n")


# ### Visualisasi Univariate & Bivariate untuk Fitur Numerik

# Visualisasi Univariate – Distribusi Nilai

# In[19]:


# Set style
sns.set(style='whitegrid')

# Fitur numerik
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Plot distribusi
plt.figure(figsize=(15, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()


# In[35]:


# Fitur numerik
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Tampilkan statistik deskriptif
df[num_cols].describe()


# Visualisasi Bivariate – Fitur Numerik vs Target Churn

# In[20]:


plt.figure(figsize=(15, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='Churn', y=col, data=df, palette='Set2')
    plt.title(f'{col} vs Churn')
plt.tight_layout()
plt.show()


# In[21]:


# Hitung rata-rata dari fitur numerik untuk masing-masing nilai Churn
df.groupby('Churn')[num_cols].mean()


# ### Korelasi dan Visualisasi Fitur Numerik

# Korelasi antar Fitur Numerik

# In[22]:


# Korelasi antar fitur numerik
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
correlation_matrix = df[numerical_features].corr()
print("Korelasi antar fitur numerik:")
print(correlation_matrix)


# In[23]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi antar fitur numerik')
plt.show()


# Korelasi antara Fitur Numerik dan Target (Churn)

# In[24]:


# Ubah kolom Churn menjadi numerik: Yes = 1, No = 0
df['Churn_numerik'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Hitung korelasi setiap fitur numerik dengan churn
for col in numerical_features:
    corr = df[col].corr(df['Churn_numerik'])
    print(f"Korelasi antara {col} dan Churn: {corr:.4f}")


# ### Visualisasi Multivariate Fitur Numerik

# In[44]:


# Langkah 13.1: Buat kategori churn numerik untuk korelasi
df['Churn_numerik'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Langkah 13.2: Hitung korelasi antara fitur numerik dan Churn_numerik
korelasi_multivariat = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_numerik']].corr()
print("Korelasi Multivariat terhadap Churn_numerik:")
print(korelasi_multivariat['Churn_numerik'].sort_values(ascending=False))


# ## Modeling

# ### Training & Evaluasi Model

# Persiapan Data

# In[ ]:


# Persiapan Data
X = df.drop(['customerID', 'Churn', 'Churn_numerik'], axis=1)
y = df['Churn_numerik']

# One-hot encoding untuk fitur kategorikal
X_encoded = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)


# Modeling Logistic Regression

# In[59]:


# Standardisasi untuk Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("=== Logistic Regression ===")
print("Akurasi:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))



# Visualisasi Model Logistic Regression

# In[60]:


# Visualisasi Confusion Matrix - Logistic Regression
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Modeling Random Forest

# In[61]:


# 5. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print("Akurasi:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# Visualisasi Model Random Forest

# In[62]:


# Visualisasi Confusion Matrix - Random Forest
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# **Hyperparameter Tuning**

# In[66]:


# === Logistic Regression Tuning ===
param_logreg = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']
}

grid_logreg = GridSearchCV(LogisticRegression(max_iter=1000), param_logreg, cv=5, scoring='accuracy')
grid_logreg.fit(X_train, y_train)

best_logreg = grid_logreg.best_estimator_
y_pred_logreg_tuned = best_logreg.predict(X_test)

print("=== Tuned Logistic Regression ===")
print("Best Params:", grid_logreg.best_params_)
print("Akurasi:", accuracy_score(y_test, y_pred_logreg_tuned))
print(classification_report(y_test, y_pred_logreg_tuned))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg_tuned))


# In[68]:


# === Random Forest Tuning ===
param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)

print("\n=== Tuned Random Forest ===")
print("Best Params:", grid_rf.best_params_)
print("Akurasi:", accuracy_score(y_test, y_pred_rf_tuned))
print(classification_report(y_test, y_pred_rf_tuned))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_tuned))


# ## Evaluation

# In[71]:


# 1. Visualisasi perbandingan metrik
metrics_logreg = {
    'Model': 'Logistic Regression',
    'Accuracy': 0.802,
    'Precision': 0.66,
    'Recall': 0.54,
    'F1 Score': 0.59
}

metrics_rf = {
    'Model': 'Random Forest',
    'Accuracy': 0.793,
    'Precision': 0.63,
    'Recall': 0.53,
    'F1 Score': 0.58
}

import pandas as pd
metrics_df = pd.DataFrame([metrics_logreg, metrics_rf])
metrics_df_melt = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10,6))
sns.barplot(data=metrics_df_melt, x='Metric', y='Score', hue='Model')
plt.title('Perbandingan Metrik Evaluasi Model')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.legend(title='Model')
plt.tight_layout()
plt.show()


# In[74]:


# 2. ROC Curve & AUC Score

# Probabilitas prediksi
y_prob_logreg = best_logreg.predict_proba(X_test)[:, 1]
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

# ROC Curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# AUC Score
auc_logreg = auc(fpr_logreg, tpr_logreg)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression vs Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

