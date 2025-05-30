# %% [markdown]
# # **Recommendation System: Rekomendasi Musik Bebasis Konten**
# ---

# %% [markdown]
# # Import Libary

# %%
# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# External Tools & API
import kagglehub
import requests

# Text Feature Extraction & Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

# Similarity-Based Recommendation
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

# %% [markdown]
# # Data Understanding

# %% [markdown]
# memahami informasi dalam data dan memnetukan kualitas dari data

# %% [markdown]
# ## Data Loading

# %%
path = kagglehub.dataset_download("devdope/200k-spotify-songs-light-dataset")
df = pd.read_csv(path + "/light_spotify_dataset.csv")

df.head()

# %% [markdown]
# mengubah nama kolom menjadi huruf kecil untuk memudahkan dalam proses analisis

# %%
df.columns = df.columns.str.lower()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# Menginvestigasi data untuk memahami karakteristik data

# %%
df.info()

# %% [markdown]
# indentifikasi jumlah fitur dataset

# %%
print("Jumlah kolom:", len(df.columns))

# %% [markdown]
# indentifikasi jumlah observasi dataset

# %%
print("Jumlah baris:", len(df))

# %% [markdown]
# identifikasin fitur numerik dan kategorik

# %%
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
print("Jumlah Fitur Numerik:", len(numerical_features))
print("Fitur Numerik:", numerical_features)

categorical_features = df.select_dtypes(include=[object]).columns.tolist()
print("Jumlah Fitur Kategorikal:", len(categorical_features))
print("Fitur Kategorikal:", categorical_features)

# %% [markdown]
# indetifikasi nilai hilang pada dataset

# %%
print("Jumlah nilai hilang:\n", df.isnull().sum())

# %% [markdown]
# indentifikasi jumlah data duplikat pada dataset

# %%
print("Jumlah nilai duplikat:", df.duplicated().sum())

# %% [markdown]
# statistik deskriptif dari fitur numerik

# %%
print(20 * "=" + " Statistik Deskriptif Fitur Numerik " + 20 * "=")
display(df[numerical_features].describe())

# %% [markdown]
# indentifikasi distribusi data dari fitur numerik

# %%
plt.figure(figsize=(20, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 4, i + 1)
    sns.histplot(df[feature], bins=20, kde=True)
    mean = df[feature].mean()
    skewness = df[feature].skew()
    plt.axvline(mean, color='red', linestyle='--', label='Mean')
    plt.title(f"Histogram {feature}")
    plt.xlabel(f"skewness: {skewness:.2f}")
    plt.legend(f"mean: {mean:.2f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# statistik deskriptif dari fitur kategorik

# %%
print(20 * "=" + " Statistik Deskriptif Fitur Kategorikal " + 20 * "=")
display(df[categorical_features].describe())

# %% [markdown]
# identifikasi nilai unik dari setiap fitur kategorik

# %%
for feature in categorical_features:
    print(10 * "=" + f" {feature} " + 10 * "=")
    print(df[feature].unique(), "\n")
    print("Jumlah unik:", df[feature].nunique(), "\n")

# %% [markdown]
# mengubah fitur genre lagu ke bentuk list

# %%
df['genre'] = df['genre'].apply(
    lambda x: x.split(',') if isinstance(x, str)
    else x if isinstance(x, list)
    else []
)

# Ambil semua genre unik
unique_genres = set()
for genres in df['genre']:
    unique_genres.update([g.strip() for g in genres])  # strip() untuk menghapus spasi ekstra

print("Jumlah genre unik:", len(unique_genres))
print("Genre unik:")
display(unique_genres)


# %% [markdown]
# identifikasi sebaran data dari fitur kategorik

# %%
top_10_dict = {
    'Artist': df['artist'].value_counts().head(10),
    'Genre': df['genre'].explode().value_counts().head(10),
    'Key': df['key'].value_counts().head(10)
}

plt.figure(figsize=(18, 10))
for i, (label, data) in enumerate(top_10_dict.items()):
    plt.subplot(1, 3, i + 1)
    sns.barplot(y=data.index, x=data.values, palette='viridis')
    plt.title(f"Top 10 {label}s")
    plt.xlabel(label)
    plt.ylabel("Jumlah Lagu")
plt.tight_layout()
plt.show()

# %%
col_to_plot = ['emotion', 'explicit']

plt.figure(figsize=(20, 7))
for i, col in enumerate(col_to_plot):
    plt.subplot(1, 2, i + 1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Jumlah Lagu per {col}")
    plt.xlabel(col)
    plt.ylabel("Jumlah Lagu")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Data Preparation

# %% [markdown]
# menghapus data dengan nilai yang hilang dan data terduplikat

# %%
df.dropna(inplace=True)
hashable_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (list, dict, set))).sum() == 0]
df.drop_duplicates(subset=hashable_columns, inplace=True)
print("Jumlah data setelah menghapus nilai duplikat:", len(df))
print("Jumlah data setelah menghapus nilai hilang:", len(df))

# %% [markdown]
# menghapus data musik dengan data genre yang bernilai 'Unknown'

# %%
df = df[~df['genre'].apply(lambda x: 'Unknown' in x)]
print("Jumlah data setelah menghapus genre 'Unknown':", len(df))

# %%
df.info()

# %% [markdown]
# menggabungkan fitur kaegorik dan memnbuat matriks TF-IDF nya

# %%
df['text'] = (df['artist'] + ' ' + df['song'] + ' ' +
              df['emotion'] + ' ' + df['key'] + ' ' +
              df['genre'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])

pd.DataFrame.sparse.from_spmatrix(
    tfidf_matrix,
    columns=vectorizer.get_feature_names_out(),
    index=df.song
)

# %% [markdown]
# mengatur skala pada fitur numerik

# %%
scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df[numerical_features])

pd.DataFrame(
    num_matrix,
    columns=numerical_features,
    index=df.song
)

# %% [markdown]
# mengabungkan matriks dari semua fitur

# %%
num_sparse = sp.csr_matrix(num_matrix)

combined_matrix = sp.hstack(
    [tfidf_matrix, num_sparse,],
    format='csr'
)

print(combined_matrix)

# %% [markdown]
# # Model Development

# %% [markdown]
# membuat model sistem rekomendasi

# %%
model = NearestNeighbors(n_neighbors=100, metric='cosine', algorithm='brute')
model.fit(combined_matrix)

def song_recommendations(song_name: str, df: pd.DataFrame, combined_matrix = combined_matrix, model = model, k=3):
    if song_name not in df['song'].values:
        raise ValueError(f"Song '{song_name}' tidak ditemukan.")
    idx = df.index[df['song'] == song_name][0]

    distances, indices = model.kneighbors(combined_matrix[idx], n_neighbors=k+1)
    rec_idxs = [i for i in indices[0] if i != idx][:k]
    rec_dists = [distances[0][list(indices[0]).index(i)] for i in rec_idxs]

    recs = df.iloc[rec_idxs].copy().reset_index(drop=True)
    similarity_scores = [1 - d for d in rec_dists]
    recs['similarity_score'] = [f"{score * 100:.2f}%" for score in similarity_scores]
    return recs[['song', 'artist', 'genre', 'similarity_score']]

# %% [markdown]
# mengambil 1 sample judul lagi dari dataset

# %%
df.sample(1, random_state=42).iloc[0]

# %% [markdown]
# melakukkan testing pada sistem rekomendasi dengan sample data

# %%
song_recommendations(df.song.sample(1, random_state=42).iloc[0], df, k=5)

# %% [markdown]
# # Evaluation

# %% [markdown]
# membuat matrik evaluasi *Normalized Discounted Cumulative Gain* (NDCG)

# %%
def ndcg_score(recommend_func, df, ground_truth, k=5):
    def dcg(relevance_scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    total_ndcg = 0
    valid_queries = 0

    for query_song, relevant_songs in ground_truth.items():
        try:
            recs = recommend_func(query_song, df, k=k)
            recommended_songs = recs['song'].tolist()
            relevance = [1 if song in relevant_songs else 0 for song in recommended_songs]
            if any(relevance):
                ideal_relevance = sorted(relevance, reverse=True)
                ndcg = dcg(relevance) / dcg(ideal_relevance)
                total_ndcg += ndcg
                valid_queries += 1
        except ValueError:
            continue

    return total_ndcg / valid_queries if valid_queries > 0 else 0.0

# %% [markdown]
# membuat data ground truth dan menghitung matrik evaluasi untuk sistem rekomendasi

# %%
ground_truth = {
    "All I Ask": [
        "Hello", "When We Were Young", "Someone Like You", "One and Only", "Remedy",
        "Chasing Pavements", "Turning Tables", "Set Fire to the Rain", "Easy On Me", "Love in the Dark"
    ]
}

score = ndcg_score(song_recommendations, df, ground_truth, k=5)
print(f"NDCG Score: {score:.4f}")


