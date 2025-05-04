# %% [markdown]
# # Proyek Pertama

# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import pkg_resources

# %% [markdown]
# ## Import Dataset

# %%
df = pd.read_csv('/home/riyan/Machine Learning Terapan/Proyek Pertama/healthcare-dataset-stroke-data.csv')
df

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# ### Gambaran Dataset

# %%
df.info()

# %%
df.describe(include='all')

# %% [markdown]
# ### Cek Data Missing

# %%
def missing_values(df):
    # Masukan nilai yang memungkinkan missing
    missing_values = ['', ' ', 'NaN', 'Nan', 'nan','NULL','Null','null','N/A','n/a', '.', ',','-','--','---', 'TIDAK ADA DATA', 'KOSONG']
    col_names = list(df.columns)
    df[col_names] = df[col_names].replace(missing_values, np.nan)

    # Hitung jumlah data yang kosong
    missing_values = df.isnull().sum()
    missing_values = pd.DataFrame(missing_values, columns=['count'])
    missing_values.reset_index(inplace=True)
    return missing_values

# %%
missing_values(df)

# %% [markdown]
# ### Cek Data Duplikat

# %%
print("Jumlah duplikat: ", df.duplicated().sum())

# %% [markdown]
# ### Distribusi Data

# %% [markdown]
# #### Distribusi Fitur Numerik

# %%
num_features = df.select_dtypes(include=[np.number])
plt.figure(figsize=(14, 10))
for i, column in enumerate(num_features.columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(df[column], bins=30, kde=True, color='blue')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Distribusi Fitur Kategorik

# %%
cat_features = df.select_dtypes(include=[object])
plt.figure(figsize=(14, 8))
for i, column in enumerate(cat_features.columns, 1):
    plt.subplot(2, 4, i)
    sns.countplot(y=df[column], palette='viridis')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Matrik Korelasi

# %%
# Heatmap korelasi untuk fitur numerik
plt.figure(figsize=(12, 10))
correlation_matrix = num_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi')
plt.show()

# %% [markdown]
# ### Distribusi Label

# %%
plt.figure(figsize=(8, 4))
sns.countplot(x='stroke', data=df, palette='viridis')
plt.title('Distribusi Variabel Target (stroke)')
plt.show()

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Hapus Data Missing

# %%
# Hapus data kosong
df = df.dropna()
df

# %% [markdown]
# ### One Hot Encoding

# %%
kategori_fitur = df.select_dtypes(include=['object']).columns
encoded = pd.get_dummies(df, columns=kategori_fitur)
encoded = encoded.astype(int)
encoded

# %% [markdown]
# ### Feature Selection

# %%
df = encoded.drop(columns=['id', 'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown', 'gender_Other'])
df
# 

# %%
df.describe(include='all')

# %% [markdown]
# ### Normalisasi Data

# %%
# Contoh data (misalnya X adalah fitur, y adalah label)
X = df.drop(columns=['stroke'])
y = df['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled = pd.concat([X_scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
X_scaled

# %%
# Simpan scaler ke file 'scaler.joblib'
joblib.dump(scaler, '/home/riyan/Machine Learning Terapan/Proyek Pertama/scaler.joblib')

# %% [markdown]
# ### SMOTE

# %%
# Terapkan SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_resampled = X_resampled.drop_duplicates()
X_resampled

# %%
plt.figure(figsize=(8, 4))
sns.countplot(x='stroke', data=X_resampled, palette='viridis')
plt.title('Distribusi Variabel Target (stroke)')
plt.show()

# %% [markdown]
# ## Model Development

# %% [markdown]
# ### Spliting Data

# %%
X = X_resampled.drop(columns=['stroke'])  # Fitur
y = X_resampled['stroke']  # Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [markdown]
# ### Training Model

# %%
# Buat model Random Forest
rf = RandomForestClassifier().fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train, y_train)

# %% [markdown]
# ## Evaluation

# %%
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results = {
        'Confusion Matrix': cm,
        'True Positive (TP)': tp,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn,
        'True Negative (TN)': tn,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    return results

# %% [markdown]
# ### Evaluation Data Testing

# %%
results = {
    'Random Forest (RF)': evaluate_model(rf, X_test, y_test),
    'K-Nearest Neighbors (KNN)': evaluate_model(knn, X_test, y_test),
}

# %%
# Buat DataFrame untuk meringkas hasil
summary_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Isi DataFrame dengan hasil
rows = []
for model_name, metrics in results.items():
    rows.append({
        'Model': model_name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score']
    })

# Konversi daftar kamus ke DataFrame
summary_df = pd.DataFrame(rows)

# Tampilkan DataFrame
print(summary_df)

# %%
# Hitung confusion matrix
y_pred = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Buat heatmap menggunakan seaborn
plt.figure(figsize=(6, 5))  # Ukuran plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# %%
# Classification Report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# %%
# Hitung confusion matrix
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Buat heatmap menggunakan seaborn
plt.figure(figsize=(6, 5))  # Ukuran plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# %%
# Classification Report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# %% [markdown]
# ### Evaluation Data Training

# %%
results = {
    'Random Forest (RF)': evaluate_model(rf, X_train, y_train),
    'K-Nearest Neighbors (KNN)': evaluate_model(knn, X_train, y_train)
}

# %%
# Buat DataFrame untuk meringkas hasil
summary_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Isi DataFrame dengan hasil
rows = []
for model_name, metrics in results.items():
    rows.append({
        'Model': model_name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score']
    })

# Konversi daftar kamus ke DataFrame
summary_df = pd.DataFrame(rows)

# Tampilkan DataFrame
print(summary_df)

# %%
# Hitung confusion matrix
y_pred = rf.predict(X_train)
conf_matrix = confusion_matrix(y_train, y_pred)

# Buat heatmap menggunakan seaborn
plt.figure(figsize=(6, 5))  # Ukuran plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# %%
# Classification Report
class_report = classification_report(y_train, y_pred)
print("\nClassification Report:")
print(class_report)

# %%
# Hitung confusion matrix
y_pred = knn.predict(X_train)
conf_matrix = confusion_matrix(y_train, y_pred)

# Buat heatmap menggunakan seaborn
plt.figure(figsize=(6, 5))  # Ukuran plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# %%
# Classification Report
class_report = classification_report(y_train, y_pred)
print("\nClassification Report:")
print(class_report)

# %% [markdown]
# ## Simpan Model

# %%
joblib.dump(rf, '/home/riyan/Machine Learning Terapan/Proyek Pertama/best_model.joblib')

# %% [markdown]
# ## Simpan Library

# %%
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

# Menyimpan nama library beserta versi yang digunakan ke file requirements.txt
with open('/home/riyan/Machine Learning Terapan/Proyek Pertama/requirements.txt', 'w') as f:
    for package in sorted(installed_packages):
        version = pkg_resources.get_distribution(package).version
        f.write(f"{package}=={version}\n")


