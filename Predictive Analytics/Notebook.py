# %% [markdown]
# # **Predictive Analytics: Prediksi Tanaman yang Cocok Ditanam Berdasarkan Kondisi Iklim**
# ---

# %% [markdown]
# ![Image Corp](https://storage.googleapis.com/kaggle-datasets-images/7367814/11736333/0dd2a975741503de21815f4405bde8b3/dataset-cover.jpg?t=2025-05-08-17-03-12)

# %% [markdown]
#   <style>
#     body {
#       font-family: Arial, sans-serif;
#       margin: 40px;
#       line-height: 1.6;
#     }
#     p {
#       text-align: justify;
#     }
#   </style>
# <p>Indonesia sebagai negara agraris memiliki potensi besar dalam sektor pertanian. Namun, produktivitas pertanian sering kali terhambat oleh kurangnya informasi yang akurat mengenai kecocokan jenis tanaman dengan kondisi lahan dan lingkungan tertentu. Banyak petani masih mengandalkan pengalaman atau perkiraan dalam menentukan tanaman yang akan ditanam, tanpa mempertimbangkan data seperti kandungan unsur hara tanah, suhu, kelembaban, pH tanah, dan curah hujan.</p>
# 
# <p>Seiring berkembangnya teknologi dan ketersediaan data, pendekatan berbasis data science dan machine learning mulai digunakan dalam mendukung pengambilan keputusan di bidang pertanian. Salah satu penerapannya adalah memprediksi jenis tanaman yang paling sesuai untuk ditanam berdasarkan kondisi lingkungan dan tanah yang tersedia.</p>
# 
# <p>Dataset ini dikembangkan untuk menjawab kebutuhan tersebut. Dengan memanfaatkan informasi seperti kandungan nitrogen (N), fosfor (P), kalium (K), suhu, kelembaban, pH tanah, dan curah hujan, model machine learning dapat dilatih untuk merekomendasikan tanaman yang paling sesuai. Hal ini bertujuan untuk membantu petani meningkatkan hasil panen, mengurangi risiko gagal panen, serta mendukung ketahanan pangan nasional.</p>
# 
# <p>Melalui proyek ini, diharapkan tercipta sistem rekomendasi yang tidak hanya bermanfaat secara praktis di lapangan, tetapi juga mendorong transformasi pertanian tradisional menuju pertanian cerdas (smart agriculture) berbasis teknologi dan data.</p>

# %% [markdown]
# # Import Library

# %% [markdown]
# Melakukan import pustakan yang akan digunakan dalam proyek

# %%
# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling - Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Modeling - Evaluation
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

# Modeling - Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Statistical Analysis
from scipy.stats import skew



# %% [markdown]
# # Data Understanding

# %% [markdown]
# Memahami informasi dalam data dan menentukan kualitas dari data

# %% [markdown]
# ## Data Loading

# %% [markdown]
# Mengunduh dataset dari [Kaggle](https://www.kaggle.com/)

# %% [markdown]
# **Informasi Dataset**
# 
# | Jenis        | Keterangan                                                                                           |
# |--------------|------------------------------------------------------------------------------------------------------|
# | Title        | Crop Recommendation Dataset                                                                          |
# | Source       | [Kaggle](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset/data)                  |
# | Maintainer   | [Atharva Ingle](https://www.kaggle.com/madhuraatmarambhagat)                                                 |
# | License      | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                                                |
# | Visibility   | Publik                                                                                               |
# | Tags         | Business, Beginner, Classification, Intermediate, Advanced, Agriculture                     |
# | Usability    | 10.00                                                                                                |
# 

# %%
import kagglehub
path = kagglehub.dataset_download("madhuraatmarambhagat/crop-recommendation-dataset")
df = pd.read_csv(path + "/Crop_recommendation.csv")
df.head()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# Menginvestigasi data untuk memahami struktur, pola, dan karakteristik dari data

# %% [markdown]
# ### Deskripsi Data

# %%
df

# %%
df.info()

# %% [markdown]
# | Nama Fitur     | Deskripsi                                    | Satuan  |
# |----------------|----------------------------------------------|---------|
# | `N`            | Kandungan Nitrogen dalam tanah               | mg/kg   |
# | `P`            | Kandungan Fosfor dalam tanah                 | mg/kg   |
# | `K`            | Kandungan Kalium dalam tanah                 | mg/kg   |
# | `temperature`  | Suhu rata-rata lingkungan                    | °C      |
# | `humidity`     | Kelembaban relatif rata-rata                 | %       |
# | `ph`           | Derajat keasaman tanah (pH)                  | -       |
# | `rainfall`     | Curah hujan rata-rata tahunan                | mm      |
# | `label`        | Jenis tanaman yang paling cocok ditanam berdasarkan kondisi tersebut | Kategori |

# %% [markdown]
# Menampilkan jumlah kolom numerik dan kategorik

# %%
print(f"jumlah kolom numerik: {df.select_dtypes(include=['int64', 'float64']).shape[1]}")
print(f"jumlah kolom kategorikal: {df.select_dtypes(include=['object']).shape[1]}")

# %% [markdown]
# Menampilkan statistik deskriptif dari data numerik untuk mengetahui sebaran dari data

# %%
df.describe()

# %% [markdown]
# Menampilkan jumlah baris dan kolom dari data

# %%
print(f"jumlah data: {df.shape[0]}")
print(f"jumlah kolom: {df.shape[1]}")

# %% [markdown]
# ### Outliers and Missing Values

# %% [markdown]
# Menampilkan jumlah nilai yang terduplikat dan hilang

# %%
print(f"Jumlah nilai yang hilang: {df.isnull().sum()}")

# %%
print(f"Jumlah nilai duplikat: {df.duplicated().sum()}")

# %% [markdown]
# Mengekplorasi sebaran data dari kolom numerik dengan visuali boxplot

# %%
df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
for col in df_numeric.columns:
    plt.subplot(3, 3, df_numeric.columns.get_loc(col) + 1)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# %% [markdown]
# Dari visualisasi boxplot diperoleh bahwa semua kolom numerik kecuali kolom N (kandungan nitrogen dalam tanah) memiliki outlier (nilai ekstrem) yang dapat memengaruhi hasil analisis dan pemodelan data. Oleh karena itu, digunakan metode interquartile range (IQR) unruk mendeteksi dan menghapus outlier pada data numerik

# %%
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

filter_outliers = ~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[filter_outliers]
print(f"Jumlah data setelah menghapus outlier: {df.shape[0]}")

# %% [markdown]
# ### Univariate Analysis

# %% [markdown]
# Membuat visualisasi histogram untuk mengetahui sebaran data setelah dilakukan filter terhadap outlier pada kolom numerik

# %%
df_numeric_filtered = df.drop(columns=['label'])
plt.figure(figsize=(12, 8))

for col in df_numeric_filtered.columns:
    plt.subplot(3, 3, df.columns.get_loc(col) + 1)
    sns.histplot(df[col], bins=20, kde=True)
    
    mean_val = df[col].mean()
    skew_val = skew(df[col].dropna())
    
    plt.axvline(mean_val, color='r', linestyle='--')
    plt.title(col)
    plt.text(mean_val, plt.gca().get_ylim()[1]*0.9, f"Mean: {mean_val:.2f}", color='r')
    plt.xlabel(f'Skewness: {skew_val:.2f}')

plt.tight_layout()
plt.show()

# %% [markdown]
# 
# | Nama Fitur    | Mean  | Skewness | Keterangan                                                        |
# |---------------|-------|----------|------------------------------------------------------------------|
# | `N`           | 54.33 | 0.35     | Distribusi sedikit miring ke kanan, relatif mendekati simetris.  |
# | `P`           | 44.85 | 0.00     | Distribusi sangat simetris dengan data tersebar cukup merata.    |
# | `K`           | 31.72 | 1.00     | Distribusi miring ke kanan dengan ekor data lebih panjang.       |
# | `temperature` | 25.84 | -0.07    | Distribusi cukup normal dengan sedikit kemiringan ke kiri.       |
# | `humidity`    | 70.11 | -0.94    | Distribusi miring ke kiri dengan ekor data lebih panjang.        |
# | `ph`          | 6.48  | -0.07    | Distribusi mendekati normal dan hampir simetris.                 |
# | `rainfall`    | 98.00 | 0.54     | Distribusi sedikit miring ke kanan.                              |
# | `label encoded`| 9.18 | 0.08     | Data numerik hasil encoding dengan distribusi cukup simetris.    |
# 

# %% [markdown]
# ### Multivariate Analysis

# %% [markdown]
# Membuat heatmap koralasi untuk mengetahui kekuatan hubungan antara pasangan data pada kolom numerik

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric_filtered.corr(), annot=True, fmt=".2f",)
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# ### Fitur Target

# %% [markdown]
# Membuat countplot kolom target untuk mengetahui sebaran data untuk masing-masing kelas

# %%
plt.figure(figsize=(12, 8))
sns.countplot(x='label', data=df)
plt.title('Distribution of Crop Labels')
plt.xticks(rotation=45)

# %% [markdown]
# Memeriksa jumlah kelas dan sebaran data dari tiap kelas

# %%
print(f"Jumlah kelas: {df.label.nunique()}")
print(f"Jumlah data per kelas:\n{df.label.value_counts()}")

# %% [markdown]
# # Data Preparation

# %% [markdown]
# Mempersiapkan data untuk membuat model machine learning

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# Mengubah label kategorik menjadi fitur numerik supaya dapat dikenali oleh model

# %%
df['label_encoded'] = df['label'].astype('category').cat.codes
df['label_encoded'].value_counts().plot(kind='bar')

# %% [markdown]
# memisah data menjadi X (variabel prediktor) dan Y (variabel respons)

# %%
X = df.drop(columns=['label', 'label_encoded'])
y = df['label_encoded']

print(f"Jumlah data: {X.shape[0]}")
print(f"Jumlah fitur: {X.shape[1]}")
print(f"Jumlah kelas: {y.nunique()}")

# %% [markdown]
# ## Train-Test Split

# %% [markdown]
# Membagi data menjadi dua bagian sebagai data latih (train) dan data uji (test)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Jumlah data keseluruhan: {df.shape[0]}")
print(f"Jumlah data latih: {X_train.shape[0]}")
print(f"Jumlah data uji: {X_test.shape[0]}")

# %% [markdown]
# ## Standarisasi

# %% [markdown]
# Menlakukan scaling pada data supaya model machine learning memiliki perporma yang lebih baik

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # Model Development

# %% [markdown]
# Melatih model klasifikasi machine learning untuk membuat prediksi

# %%
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []
n_classes = len(np.unique(y_train))

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Inisialisasi y_proba
    y_proba = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
    else:
        try:
            y_score = model.decision_function(X_test_scaled)
            if len(y_score.shape) == 1:
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
                y_proba = np.vstack([1 - y_score, y_score]).T
            else:
                y_proba = y_score
        except:
            y_proba = None

    if y_proba is not None and len(y_proba.shape) == 2:
        try:
            roc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except:
            roc = np.nan
    else:
        roc = np.nan

    # Tentukan apakah overfitting berdasarkan selisih > 0.1
    overfit_status = "Overfitting" if train_acc - test_acc > 0.05 else "No"

    results.append({
        'Model': name,
        'Train Accuracy': round(train_acc, 3),
        'Test Accuracy': round(test_acc, 3),
        'ROC AUC': round(roc, 3) if not np.isnan(roc) else np.nan,
        'Overfitting': overfit_status
    })

# Tampilkan hasil
results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)

# %% [markdown]
# # Evaluasi

# %% [markdown]
# Mengevaluasi model klasifikasi stelah dilakukan pelatihan
# 
# Evaluasi model menggunakan metrik ROC AUC karena metrik ini dapat mengukur **kemampuan model dalam membedakan antara kelas positif dan negatif secara menyeluruh, tanpa bergantung pada satu nilai threshold tertentu**. ROC AUC mempertimbangkan seluruh kemungkinan ambang batas klasifikasi dan memberikan gambaran umum tentang performa model dalam hal trade-off antara sensitivitas (*True Positive Rate*) dan spesifisitas (*1 - False Positive Rate*).

# %%
results_sorted = results_df.sort_values('ROC AUC', ascending=False)
results_sorted

# %% [markdown]
# Interpretasi:
# - **Random Forest** dan **Decision Tree** mencapai akurasi sempurna (1.000) pada data pelatihan, menunjukkan model mampu menangkap pola dengan sangat baik. Namun, tidak terjadi overfitting karena akurasi pada data uji tetap sangat tinggi (di atas 0.99), dan nilai ROC AUC mendekati 1.000.
# - **KNN** memiliki akurasi train dan test yang lebih rendah dibanding model lain, namun tetap tinggi dan seimbang, menunjukkan generalisasi yang baik. Nilai ROC AUC juga sangat tinggi (0.992), menandakan bahwa model ini tetap mampu membedakan kelas dengan baik.
# 
# Kesimpulan:
# Semua model menunjukkan performa sangat baik tanpa indikasi overfitting. Random Forest dan Decision Tree tampil sebagai model terbaik dengan akurasi tinggi dan stabilitas yang sangat baik. KNN memberikan alternatif yang lebih ringan namun tetap efektif.

# %% [markdown]
# Membuat visualisasi performa dari model

# %%
plt.figure(figsize=(12, 8))
sns.barplot(y='Model', x='ROC AUC', data=results_sorted, palette='viridis')
plt.title('Model Comparison - ROC AUC')


