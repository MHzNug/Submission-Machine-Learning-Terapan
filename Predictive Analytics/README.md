# Laporan Proyek Machine Learning - Muhammad Husni Zahran Nugrahanto

## Domain Proyek
Proyek ini memiliki domain di bidang pertanian yang berfokus kepada prediksi kecocokan tanaman terhadap kondisi lahan dan lingkungan tertentu.

![Foto Pertanian](https://i.ibb.co/GfKgtch8/dataset-cover.jpg)

### Latar Belakang
Indonesian merupakan negara agraris yang memiliki ketergantungan tinggi pada sektor pertanian sebagai penopang ekonomi dan ketahanan pangan. Namun, produktivitas dari sektor ini seringkali terhambat karena ketidaksesuaian antara jenis tanaman pertanian yang ditanam dengan kondisi lingkungan setempat, seperti kandungan unsur hara dalam tanah, suhu, pH tanah, dan curah hujan. ketidak sesuaian ini dapat menyebabkan hasil panen yang rendah dan kerugian bagi petani.

Sebagai contoh, studi kasus pada tanaman jagung di Kabupaten Malang menunjukkan bahwa produktivitas jagung dipengaruhi secara signifikan oleh faktor suhu dan intensitas curah hujan [[1](https://journal.ipb.ac.id/index.php/JIPI/article/view/51574/30158)]. Selain itu, penelitian lain mengenai pertumbuhan tanaman secara umum juga menemukan bahwa kondisi lingkungan seperti suhu, kelembapan udara, intensitas cahaya, dan curah hujan turut menentukan baik atau buruknya perkembangan tanaman [[2](https://journal.uii.ac.id/Snati/article/view/3126/2859)]. Oleh karena itu, pemilihan jenis tanaman yang sesuai dengan karakteristik lingkungan setempat menjadi krusial untuk meningkatkan produktivitas pertanian dan mengurangi risiko kerugian bagi petani.

### Permasalahan dan Solusi yang Ditawarkan
Permasalahan utama yang dihadapi adalah kurangnya sistem yang dapat memberikan rekomendasi tanaman secara akurat berdasarkan variabel lingkungan seperti kandungan nitrogen (N), fosfor (P), kalium (K), suhu, kelembaban, pH tanah, dan curah hujan. Selain itu, Kebanyakan petani masih mengandalkan pengetahuan tradisional atau pengalaman pribadi, yang mungkin tidak selalu sesuai dengan kondisi aktual lahan mereka. Oleh karena itu, diperlukan solusi berbasis data yang mampu untuk mengintegrasikan variabel-variabel lingkungan  dengan metode analitik guna menghasilkan rekomendasi tanaman yang lebih akurat. 

Masalah ini penting untuk diselesaikan karena:
- **Meningkatkan Produktivitas:** \
Dengan memilih tanaman yang sesuai, hasil panen dapat meningkat secara signifikan.
- **Efisiensi Sumber Daya:** \
Penggunaan pupuk dan air dapat dioptimalkan sesuai kebutuhan tanaman yang direkomendasikan.
- **Katahanan Pangan:** \
Dengan hasil panen yang lebih baik, ketahanan pangan nasional dapat terjaga.
- **Pengurangan Risiko Kerugian:** \
Petani dapat mengurangi risiko kerugian akibat gagal panen karena ketidaksesuaian tanaman dengan kondisi lingkungan.

Dengan demikian, dilakukan pengembangan model *machine learning* (ML) yang dapat mempelajari pola dari data historis sehingga dapat memberikan rekomendasi yang lebih akurat dan adaptif  terhadap perubahan kondisi lingkungan.

## Business Understanding
Pengembangan model prediksi dalam memilih tanaman yang tepat dengan kondisi lingkungan setempat dapat memberikan manfaat bagi berbagi pihak yang salah satunya adalah petani. Dengan solusi berbasis data ini, diharapkan dapat memberikan rekomendasi tanaman yang optimal terhadap kondisi lingkungan setempat. Dengan demikian, petani dapat menurunkan risiko kerugian serta meningkatkan efisiensi sumber daya yang dibutuhkan.

### Problem Statements
Berdasarkan latar belakang, berikut adalah rincian masalah yang dapat dirumuskan dalam proyek ini:
- Bagaimana cara memilih tanaman yang cocok berdasarkan variabilitas kondisi lahan (N, P, K, pH) dan iklim (suhu, kelembapan, dan urah hujan)?
- Bagaimana cara menurunkan risiko gagal panen dampak dari kesalahan dalam memilih tanaman?

### Goals
Tujuan dari proyek ini adalah:
- Mengembangkan serta melatih model ML untuk merekomendasikan tanaman sesuai dengan kondisi lingkungan setempat.
- Menargetkan nilai matrik evaluasi model ROC-AUC minimal 0.80 pada data uji.

### Solution statements
Untuk mencapai tujuan proyek ini, akan dilakukan pengembangan model ML dengan tiga pendekatan solusi yang dievaluasi dengan metrik ROC-AUC serta akurasi dari data pelatihan (train) dan pengujian (test) untuk mendeteksi potensi overfitting pada model:
- Model yang digunakan:
    - Decision Tree Classifier adalah model yang mirip dengan bagan alur dengan serangakian pertanyaan "ya/tidak" untuk membagi data menjadi beberapa kelompok hingga mencapai hasil keputusan di ujung setiap cabang[[3](https://medium.com/@MrBam44/decision-trees-91f61a42c724)].
    - Random Forest Classifier adalah pengembangan dari algoritma decision tree, model ini membangun banyak decision tree  secara acak (dengan subset data dan fitur yang berbeda) kemudian mengabungkan hasil prediksi sebagai "suara mayoritas" sebagai hasil akhirnya[[4](https://ishanjainoffical.medium.com/understanding-random-forest-algorithm-with-python-code-ae6fb0e34938)].
    - K-Nearest Neighbors (KNN) adalah metode yang mengklasifikasi data baru dengan melihat $k$ tetangga terdekat pada ruang fitur dan menetapkan kelas berdasarkan kelas yang paling banyak muncul diantara tetangga tersebut[[5](https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4)]. 
- Prosedur evaluasi:
    - Membagi dataset menjadi set pelatihan dan pengujian dengan proporsi 80% data pelatihan dan 20% data pengujian.
    - Melatih model ML dengan data pelatihan.
    - Pengukuran kinerja dengan menghitung akurasi pada data pelatihan dan pengujian dan menghitung ROC-AUC pada data pelatihan.
    - Membandingkan selisih nilai akurasi antara data pelatihan dengan data pengujian jika selisih akurasi $>0,05$ model dianggap memiliki potensi overfitting.
    - Memilih model  dengan keseimbangan optimal (ROC-AUC tinggi) dan tidak memiliki potensi overfitting.

## Data Understanding
Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com) yang berisi informasi mengenai jenis tanaman yang cocok untuk dibudidayakan sesuai dengan kondisi lingkungan setempat. Dataset terdiri dari 22000 observasi tanpa nilai yang hilang dan data duplikat dengan 8 fitur yang mencakup 7 fitur input numerik dan 1 fitur target output berupa jenis tanaman yang sesuai. berikut adalah ringkasan informasi dari dataset:
| Jenis        | Keterangan                                                                                           |
|--------------|------------------------------------------------------------------------------------------------------|
| Title        | Crop Recommendation Dataset                                                                          |
| Source       | [Kaggle](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset/data)                  |
| Maintainer   | [Atharva Ingle](https://www.kaggle.com/madhuraatmarambhagat)                                                 |
| License      | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                                                                |
| Visibility   | Publik                                                                                               |
| Tags         | Business, Beginner, Classification, Intermediate, Advanced, Agriculture                     |
| Usability    | 10.00                                                                                                |

Berikut adalah cuplikan dari datset:
| N  | P  | K  | Temperature | Humidity | pH  | Rainfall | Label |
|----|----|----|-------------|----------|-----|-----------|--------|
| 90 | 42 | 43 | 20.87       | 82.0     | 6.50 | 202.9     | rice   |
| 85 | 58 | 41 | 21.77       | 80.3     | 7.00 | 226.4     | rice   |
| 60 | 55 | 44 | 23.00       | 82.0     | 7.20 | 263.9     | rice   |
| 74 | 35 | 40 | 26.65       | 80.0     | 6.90 | 242.7     | rice   |
| 78 | 42 | 42 | 20.87       | 82.0     | 6.50 | 202.9     | rice   |

### Fitur pada Corp Recommendation dataset
| Nama Fitur     | Deskripsi                                    | Satuan  |
|----------------|----------------------------------------------|---------|
| `N`            | Kandungan Nitrogen dalam tanah               | mg/kg   |
| `P`            | Kandungan Fosfor dalam tanah                 | mg/kg   |
| `K`            | Kandungan Kalium dalam tanah                 | mg/kg   |
| `temperature`  | Suhu rata-rata lingkungan                    | °C      |
| `humidity`     | Kelembaban relatif rata-rata                 | %       |
| `ph`           | Derajat keasaman tanah (pH)                  | -       |
| `rainfall`     | Curah hujan rata-rata tahunan                | mm      |
| `label`        | Jenis tanaman yang paling cocok ditanam berdasarkan kondisi tersebut | Kategori |

### *Explooratory Data Analysis* (EDA)
#### Deteksi Nilai Ekstream (*Outlier*) dengan Visualisasi Boxplot
![Boxplot Fitur Numerik](boxplot.png)
Visulisasi *boxplot* adalah salah satu metode analisis univariat yang digunakan untuk melihat sebaran data numerik. Dari visualisasi *boxplot* diperoleh bahwa fitur `p`, `K`, `Temperature`, `humidity`, `pH`, dan `rainfall` memiliki  (*outlier*). Oleh karena itu, dilakukan penanganan *outlier* dengan metode *interquartile range* (IQR).

#### Penanganan Nilai Ekstream (*Outlier*) dengan metode 
metode IQR adalah metode yang digunakan untuk mengatasi *outlier* dengan mebuang/menghapus nilai yang berada diluar batas atas dan batas bawah. Cara mengidetifikasi *outlier* adalah dengan mengurutkan nilai numerik dan membaginya menjadi empat bagian sama rata. Titik di akhir bagian pertama disebut $Q1$ (kuartil pertama), dan titik di akhir bagian ketiga disebut $Q3$ (kuartil ketiga). Jarak antara $Q1$ dan $Q3$ disebut IQR. secara matematis dapat dituliskan sebagai berikut:
$$IQR=Q3−Q1$$
$$Batas Bawah=Q1−1,5×IQR$$
$$Batas Atas=Q3+1,5×IQR$$
Semua data yang nilainya kurang dari Batas Bawah atau lebih dari Batas Atas dianggap outlier dan dihapus[[6](https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d)].

Penanganan *outlier* diperlukan karena *outlier* dapat menyebabkan bias pada batas-batas klasifikasi terutama pada algoritma berbasis jarak. Selain itu, dengan mengapus *outlier* dapat mengurangi *noise* dan meningkatkan performa dari model klassifikasi. 

Setelah dilakukan penanganan pada *outlier* data jumlah observasi data berkurang menjadi 1768 observasi dan diperoleh sebaran data dengan visualisasi histogram sebagai berikut:
 ![Histogram Fitur Numerik (Clean)](Histogram.png)
 Interpretasi:
 
| Nama Fitur    | Mean  | Skewness | Keterangan                                                        |
|---------------|-------|----------|------------------------------------------------------------------|
| `N`           | 54.33 | 0.35     | Distribusi sedikit miring ke kanan, relatif mendekati simetris.  |
| `P`           | 44.85 | 0.00     | Distribusi sangat simetris dengan data tersebar cukup merata.    |
| `K`           | 31.72 | 1.00     | Distribusi miring ke kanan dengan ekor data lebih panjang.       |
| `temperature` | 25.84 | -0.07    | Distribusi cukup normal dengan sedikit kemiringan ke kiri.       |
| `humidity`    | 70.11 | -0.94    | Distribusi miring ke kiri dengan ekor data lebih panjang.        |
| `ph`          | 6.48  | -0.07    | Distribusi mendekati normal dan hampir simetris.                 |
| `rainfall`    | 98.00 | 0.54     | Distribusi sedikit miring ke kanan.                              |
| `label encoded`| 9.18 | 0.08     | Data numerik hasil encoding dengan distribusi cukup simetris.    |

#### Multivariate Analysis
![Heatmap Korelasi](Heatmap.png)
Visualisasi Heatmap digunakan untuk mengambarkan korelasi antara pasangan fitur numerik. Dari visualisasi diperoleh bahwa tidak ada fitur yang memiliki korelasi yang kuat (mendekati 1 atau -1). Fitur `humidity` memiliki hubungan negatif yang cukup kuat dengan `P` dan hubungan positif dengan `N` dan `temperature`. Fitur `N` dan `K` juga memiliki hubungan positif yang sedang. Sedangkan variabel yang lainnya secara umum memiliki korelasi hubungan yang rendah antara satu sama lain.

### Fitur Target
![Count Plot Target](Coutplot.png)
Visualisasi *coutplot* menampilkan distribusi dari fitur kategorik target. Dari visualisasi *coutplot*, diperoleh bahwa fitur target memiliki 20 kelas dengan jumlah observasi yang bervariasi. Sebagian besar kelas memilki 100 observasi, tapi beberapa kelas seperti rice (32), papaya (54), dan chickpea (58) jauh lebih sedikit jika dibandingkan dengan kelas yang lain. Hal ini dapat diartikan bahwa jumlah observasi pada setiap kelasnya tidak seimbang sehingga dapat mempengaruhi performa model.

## Data Preparation
Berikut adalah tahapan dalam menyiapkan data secara berurutan:
- **Spliting data** \
Membagi dataset menjadi dua bagian sebagai data latih (*train*) dan data uji (*test*). Pembagian dataset bertujuan untuk melatih dan mengevaluasi kinerja dari model. Pada proyek ini, digunakan proporsi *train* sebesar $80%$ untuk melatih model dan *test* sebesar *20%* untuk mengevaluasi kinerja dari mopdel.
- **Standarisasi data** \
mengubah skals nilai fitur numerik dengan tujuan supaya fitur numerik memiliki $rata-rata(\mu)=0$ dan $simpangan baku(\sigma)=1$ [[7](https://medium.com/@onersarpnalcin/standardscaler-vs-minmaxscaler-vs-robustscaler-which-one-to-use-for-your-next-ml-project-ae5b44f571b9)]. Secara matematis dapat dituliskan, sebagai berikut:
$$Z=\frac{X-\mu}{\sigma}$$ 
Keterangan:
- $Z$ : Nilai hasil standarisasi
- $X$ : Nilai asli
- $\mu$ : rata-rata dari seluruh nilai pada fitur tersebut
- $\sigma$ : simpangan baku dari fitur tersebut
Tujuan dari tahapan ini adalah supaya setiap fitur memiliki kontribusi yang setara saat melatih model ML sehingga dapat meningkatkan performa dari model ML.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

## Refrerensi
[[1](https://journal.ipb.ac.id/index.php/JIPI/article/view/51574/30158)] N. Herlina dan A. Prasetyorini, "Pengaruh Perubahan Iklim pada Musim Tanam dan Produktivitas Jagung (Zea mays L.) di Kabupaten Malang," *Jurnal Ilmu Pertanian Indonesia (JIPI)*, vol. 25, no. 1, pp. 118–128, Jan. 2020, doi: [10.18343/jipi.25.1.118.](10.18343/jipi.25.1.118.) \
[[2](https://journal.uii.ac.id/Snati/article/view/3126/2859)] D. Heksaputra, Y. Azani, Z. Naimah, dan L. Iswari, "Penentuan Pengaruh Iklim Terhadap Pertumbuhan Tanaman dengan Naïve Bayes," *Seminar Nasional Aplikasi Teknologi Informasi (SNATI)*, Yogyakarta, vol. N-34, pp. N-34–N-39, Jun. 2013, ISSN: 1907-5022. \
[[3](https://medium.com/@MrBam44/decision-trees-91f61a42c724)] S. Koli, “Decision Trees: A Complete Introduction With Examples,” Medium, Feb. 27, 2023. [Online]. https://medium.com/@MrBam44/decision-trees-91f61a42c724. \
[[4](https://ishanjainoffical.medium.com/understanding-random-forest-algorithm-with-python-code-ae6fb0e34938)] Data Science & Beyond, “Understanding Random Forest Algorithm with Python Code,” Medium, Oct. 2, 2023. [Online]. https://ishanjainoffical.medium.com/understanding-random-forest-algorithm-with-python-code-ae6fb0e34938. \
[[5](https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4)] A. Christopher, “K-Nearest Neighbor,” The Startup, Medium, Feb. 2, 2021. [Online]. https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4. \
[[6](https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d)] P. Patel, “Outlier Detection and Removal using the IQR Method,” Medium, Dec. 14, 2021. [Online]. Available: https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d. \
[[7](https://medium.com/@onersarpnalcin/standardscaler-vs-minmaxscaler-vs-robustscaler-which-one-to-use-for-your-next-ml-project-ae5b44f571b9)] O. S. Nalçin, "StandardScaler vs MinMaxScaler vs RobustScaler — Which one to use for your next ML project?," Medium, May 4, 2023. [Online]. https://medium.com/@onersarpnalcin/standardscaler-vs-minmaxscaler-vs-robustscaler-which-one-to-use-for-your-next-ml-project-ae5b44f571b9.