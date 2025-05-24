# Laporan Proyek Machine Learning - Muhammad Husni Zahran Nugrahanto

## Domain Proyek
Proyek ini memiliki domain di bidang pertanian yang berfokus kepada prediksi kecocokan tanaman terhadap kondisi lahan dan lingkungan tertentu.

![Foto Pertanian](https://storage.googleapis.com/kaggle-datasets-images/7367814/11736333/0dd2a975741503de21815f4405bde8b3/dataset-cover.jpg?t=2025-05-08-17-03-12)

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
Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/madhuraatmarambhagat) yang berisi informasi mengenai jenis tanaman yang cocok untuk dibudidayakan sesuai dengan kondisi lingkungan setempat. Dataset terdiri dari 22000 observasi tanpa nilai yang hilang dan data duplikat dengan 8 fitur yang mencakup 7 fitur input numerik dan 1 fitur target output berupa jenis tanaman yang sesuai. berikut adalah ringkasan informasi dari dataset:
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

### Deteksi dan menangani Nilai Ekstream dengan Visualisasi Boxplot
![Boxplot Fitur Numerik](boxplot.png)
Dari visualisasi boxplot diperoleh bahwa fiitur `p`, `K`, `Temperature`, `humidity`, `pH`, dan `rainfall` memiliki nilai ekstream. Oleh karena itu, dilakukan penanganan missing value dengan metode IQR

### 

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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
[[5](https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4)] A. Christopher, “K-Nearest Neighbor,” The Startup, Medium, Feb. 2, 2021. [Online]. https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4.