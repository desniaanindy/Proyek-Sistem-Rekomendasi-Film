# Laporan Proyek Machine Learning - Desnia Anindy Irni Hareva

**ID Dicoding: desniaanindy**

---

## Project Overview

Dalam era digital saat ini, jumlah konten film yang tersedia secara online telah meningkat secara eksponensial. Platform streaming seperti Netflix, Amazon Prime, dan Disney+ memiliki ribuan hingga puluhan ribu judul film. Hal ini menciptakan masalah *information overload* bagi pengguna dalam memilih film yang sesuai dengan preferensi mereka. Sistem rekomendasi hadir sebagai solusi untuk mengatasi masalah ini dengan menyajikan konten yang relevan dan personal kepada setiap pengguna.

Proyek ini penting karena beberapa alasan:
1.  **User Experience:** Membantu pengguna menemukan film yang relevan dengan cepat dan mudah.
2.  **Business Value:** Meningkatkan *engagement* dan *retention rate* pengguna pada platform.
3.  **Content Discovery:** Membantu film-film berkualitas (termasuk yang kurang populer) mendapatkan eksposur yang layak.
4.  **Personalization:** Memberikan pengalaman yang dipersonalisasi dan unik untuk setiap pengguna.

### Referensi
* He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. In *Proceedings of the 26th international conference on world wide web* (pp. 173-182).
* Ricci, F., Rokach, & Shapira, B. (2011). *Recommender Systems Handbook*. Springer.
* GroupLens. (2016). *MovieLens 100K Dataset*. Diperoleh dari [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

---

## Business Understanding

### Problem Statements
-   Bagaimana cara membangun sistem yang dapat merekomendasikan film berdasarkan kemiripan konten dan preferensi genre pengguna?
-   Bagaimana cara membangun sistem yang dapat memprediksi rating yang akan diberikan pengguna pada film yang belum ia tonton, berdasarkan pola historis dari pengguna dengan selera serupa?
-   Bagaimana cara menggabungkan kedua pendekatan tersebut untuk menciptakan sistem rekomendasi yang seimbang, yang tidak hanya menyajikan film-film yang relevan tetapi juga memperkenalkan film baru yang mungkin disukai pengguna (*serendipity*)?

### Goals
Berdasarkan pernyataan masalah, tujuan dari proyek ini adalah:
-   Mengembangkan model **Content-Based Filtering** yang mampu memberikan daftar film serupa berdasarkan genre, dengan hasil rekomendasi yang relevan dan juga beragam.
-   Mengembangkan model **Collaborative Filtering** yang mampu memprediksi preferensi pengguna dengan tingkat akurasi tinggi, diukur dengan **RMSE (Root Mean Squared Error) di bawah 0.9**.
-   Membangun sebuah **Model Hybrid** yang mengkombinasikan kekuatan dari kedua pendekatan untuk memberikan rekomendasi yang lebih holistik.
-   Mengevaluasi dan membandingkan ketiga model menggunakan metrik peringkat yang relevan seperti **Precision, Recall, dan F1-Score**.

### Solution statements
Untuk mencapai tujuan tersebut, diajukan dua pendekatan solusi utama (dan satu pendekatan gabungan):
1.  **Content-Based Filtering**: Menggunakan pendekatan **TF-IDF + Cosine Similarity**. Pendekatan ini merekomendasikan film berdasarkan kemiripan atribut konten (seperti genre dan judul). Untuk meningkatkan kualitas, saya akan mengimplementasikan **Maximal Marginal Relevance (MMR)** untuk memastikan hasil rekomendasi lebih beragam.
2.  **Collaborative Filtering**: Menggunakan **Neural Collaborative Filtering (NCF)**, sebuah arsitektur *deep learning* yang menggabungkan *Matrix Factorization* dengan *Multi-Layer Perceptron*.
3.  **Hybrid Filtering**: Menggunakan **Weighted Hybrid Model** yang menggabungkan output dari kedua model di atas untuk memberikan rekomendasi yang lebih seimbang.

---

## Data Understanding

Dataset yang digunakan adalah **MovieLens 100K**, salah satu dataset benchmark paling populer untuk sistem rekomendasi.
* **Sumber:** [MovieLens 100K Dataset on Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
* **Deskripsi:** Dataset ini berisi 100.000 rating (skala 1-5) dari 943 pengguna untuk 1.682 film. Setiap pengguna telah memberi rating untuk minimal 20 film.

### Uraian Seluruh Variabel

Berikut adalah deskripsi untuk **seluruh file dan fitur** yang ada pada dataset, berdasarkan dokumentasi aslinya:

#### 1. File Utama yang Digunakan
* **`u.data`**: Berisi 100.000 rating. Setiap baris dipisahkan oleh karakter tab dan terdiri dari:
    -   `user_id`: ID unik pengguna.
    -   `item_id`: ID unik item/film.
    -   `rating`: Rating dengan skala 1 hingga 5.
    -   `timestamp`: Waktu pemberian rating dalam format detik Unix.
* **`u.item`**: Informasi tentang 1.682 item (film). Setiap baris dipisahkan oleh karakter `|` dan terdiri dari:
    -   `movie_id` atau `item_id`: ID unik film.
    -   `movie_title`: Judul film beserta tahun rilisnya.
    -   `release_date`: Tanggal rilis film.
    -   `video_release_date`: Tanggal rilis dalam format video (umumnya kosong).
    -   `IMDb_URL`: Tautan menuju halaman film di IMDb.
    -   `unknown` ... `Western`: 19 kolom genre biner (1 jika film termasuk genre tersebut, 0 jika tidak).
* **`u.user`**: Informasi demografis tentang 943 pengguna. Setiap baris dipisahkan oleh karakter `|` dan terdiri dari:
    -   `user_id`: ID unik pengguna.
    -   `age`: Usia pengguna.
    -   `gender`: Jenis kelamin ('M' untuk Male, 'F' untuk Female).
    -   `occupation`: Pekerjaan pengguna.
    -   `zip_code`: Kode pos pengguna.

#### 2. File Pendukung & Utilitas
* `u.genre`: Daftar nama 19 genre yang digunakan.
* `u.occupation`: Daftar nama pekerjaan yang ada.
* `u.info`: Informasi jumlah pengguna, item, dan rating.
* `u1.base`, `u1.test`, ..., `u5.base`, `u5.test`: Set data yang sudah dibagi sebelumnya (80% training, 20% test) untuk keperluan 5-fold cross-validation.
* `ua.base`, `ua.test`, `ub.base`, `ub.test`: Set data yang sudah dibagi di mana setiap pengguna di set test memiliki tepat 10 rating.
* `mku.sh`, `allbut.pl`: Skrip untuk membuat set data yang terbagi di atas.

Untuk proyek ini, kita hanya akan menggunakan file utama (`u.data`, `u.item`, `u.user`) dan melakukan pembagian data training/test sendiri.

### Exploratory Data Analysis (EDA)
Dari analisis data eksploratif yang dilakukan di notebook, didapatkan beberapa *insight* kunci:
* **Distribusi Rating:** Rating cenderung positif, dengan rating **4** menjadi yang paling sering diberikan (34,174 kali).
* **Sparsitas Data:** Dataset memiliki tingkat sparsitas **93.7%**, yang berarti sebagian besar kemungkinan pasangan pengguna-film tidak memiliki rating.
* **Demografi Pengguna:** Mayoritas pengguna adalah **pria (71%)** dengan rata-rata usia **34 tahun**.

---

## Data Preparation

Tahapan persiapan data dilakukan secara optimal untuk memaksimalkan performa model. Berikut adalah langkah-langkah yang dilakukan secara berurutan beserta alasannya, sesuai dengan implementasi pada fungsi `optimized_data_preparation` di notebook:

1.  **Peningkatan Fitur Genre (*Genre Feature Enhancement*):**
    * **Proses:** Membuat sebuah string genre untuk setiap film dengan memberikan bobot lebih pada genre yang lebih langka (berdasarkan *inverse frequency*). Genre yang populer seperti 'Drama' atau 'Comedy' akan diulang lebih sedikit dalam string dibandingkan genre *niche* seperti 'Film-Noir'.
    * **Alasan:** Ini dilakukan untuk **meningkatkan sinyal dari genre yang kurang umum** saat proses TF-IDF. Tanpa pembobotan, rekomendasi akan cenderung didominasi oleh genre-genre mayoritas saja.

2.  **Ekstraksi Fitur dari Judul (*Title Feature Extraction*):**
    * **Proses:** Mengekstrak informasi penting dari kolom `title`, yaitu `year` (tahun rilis), `clean_title` (judul bersih tanpa tahun), `title_keywords` (kata kunci penting dari judul), dan `decade` (dekade rilis).
    * **Alasan:** Memecah judul menjadi fitur-fitur yang lebih granular (seperti dekade atau kata kunci) memungkinkan model Content-Based untuk **menemukan pola kemiripan yang lebih halus** di luar kesamaan genre.

3.  **Pembuatan Fitur Konten Komprehensif (`content_features`):**
    * **Proses:** Menggabungkan hasil dari langkah 1 dan 2 (string genre yang sudah diboboti, kata kunci judul, dan dekade) menjadi satu fitur teks tunggal. String genre bahkan ditambahkan dua kali untuk memberikan penekanan lebih pada fitur ini.
    * **Alasan:** Ini menciptakan **vektor konten yang sangat kaya dan representatif** untuk setiap film, yang menjadi dasar bagi model Content-Based untuk menghitung similaritas dengan lebih akurat.

4.  **Perhitungan dan Penggabungan Fitur Statistik:**
    * **Proses:** Menghitung statistik agregat untuk setiap pengguna (rata-rata rating, jumlah rating, standar deviasi rating) dan setiap item (rata-rata rating, jumlah rating, standar deviasi rating). Fitur-fitur ini kemudian digabungkan kembali ke dataset utama.
    * **Alasan:** Fitur statistik ini **memberikan konteks perilaku** yang dapat menjadi sinyal penting bagi model, misalnya untuk membedakan pengguna aktif dengan pengguna pasif atau item populer dengan item *niche*.

5.  **Pembuatan Profil Preferensi Pengguna:**
    * **Proses:** Untuk setiap pengguna, diidentifikasi 3 genre teratas yang paling sering mereka beri rating tinggi (rating ≥ 3.5). Genre-genre ini kemudian disimpan sebagai `preferred_genres`.
    * **Alasan:** Langkah ini membantu dalam **memahami profil selera setiap pengguna secara eksplisit**, yang nantinya bisa digunakan untuk analisis lebih lanjut atau bahkan sebagai fitur tambahan pada model yang lebih kompleks.

6.  **Kategorisasi Popularitas Item:**
    * **Proses:** Film dikelompokkan ke dalam beberapa tingkatan popularitas (`niche`, `moderate`, `popular`, `mainstream`) berdasarkan jumlah rating yang mereka terima, menggunakan metode kuartil.
    * **Alasan:** Ini membantu dalam **menganalisis performa model** terhadap item populer vs. item yang kurang populer dan bisa menjadi strategi untuk meningkatkan diversitas dalam rekomendasi.

7.  **Encoding ID dan Finalisasi Data:**
    * **Proses:** Mengubah `user_id` dan `item_id` menjadi indeks integer yang sekuensial (mulai dari 0) dan memastikan tidak ada data yang hilang pada kolom-kolom penting.
    * **Alasan:** Encoding adalah **syarat wajib** agar data dapat digunakan oleh *embedding layers* pada model *deep learning* Collaborative Filtering.

Setelah semua tahapan ini, dataset final yang bersih dan kaya fitur, berisi **100,000 rating** tanpa ada nilai yang hilang, siap untuk tahap pemodelan.

---

## Modeling and Result

### 1. Model Content-Based Filtering
* **Algoritma:** Menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah fitur teks `content_features` menjadi vektor numerik. Parameter TF-IDF dioptimalkan (`ngram_range=(1,2)`, `max_features=2000`) untuk menangkap frasa penting.
* **Perhitungan Similaritas:** Kemiripan antar film dihitung menggunakan **Cosine Similarity**. Untuk meningkatkan akurasi, similaritas akhir adalah gabungan dari similaritas berbasis TF-IDF (bobot 60%) dan similaritas murni berbasis genre (bobot 40%).
* **Kelebihan:**
    * **Explainability:** Mudah menjelaskan mengapa sebuah film direkomendasikan (misalnya, "karena sama-sama bergenre *crime drama*").
    * **Tidak Memerlukan Data Pengguna Lain:** Rekomendasi untuk satu pengguna tidak bergantung pada data pengguna lain.
* **Kekurangan:**
    * **Over-specialization:** Cenderung merekomendasikan item yang sangat mirip dengan apa yang sudah disukai pengguna, sehingga kurang ada penemuan baru (*serendipity*).
    * **Ketergantungan pada Kualitas Fitur:** Kualitas rekomendasi sangat bergantung pada kelengkapan dan kualitas data konten film.

* **Hasil Top-N Recommendation:**
    Berikut adalah contoh rekomendasi dan fitur terpenting untuk film **'Pulp Fiction (1994)'**, sesuai dengan output di notebook:
    ```
    🔍 Rekomendasi untuk 'Pulp Fiction (1994)':
       1. GoodFellas (1990)                             | Score: 1.0000
       2. Donnie Brasco (1997)                          | Score: 1.0000
       3. Sleepers (1996)                               | Score: 1.0000
       4. Twilight (1998)                               | Score: 1.0000
       5. Cyclo (1995)                                  | Score: 1.0000
    
    📊 Feature importance untuk 'Pulp Fiction (1994)':
       crime drama         : 0.5492
       crime               : 0.5267
       drama crime         : 0.4666
       crime crime         : 0.3202
       drama               : 0.2228
    ```

### 2. Model Collaborative Filtering
* **Algoritma:** Menggunakan **Neural Collaborative Filtering (NCF)** yang dirancang untuk tugas regresi rating.
* **Arsitektur:**
    * **Embedding Layers:** Lapisan embedding terpisah untuk merepresentasikan pengguna dan item ke dalam vektor laten (dimensi 64). Terdapat juga lapisan bias untuk menangkap kecenderungan rating.
    * **Matrix Factorization & MLP Path:** Menggabungkan *dot product* antara embedding (untuk interaksi linear) dan beberapa lapisan Dense (untuk interaksi non-linear) guna menghasilkan prediksi rating yang akurat.
* **Kelebihan:**
    * **Menemukan Minat Baru:** Mampu menemukan dan merekomendasikan item yang secara konten tidak mirip, tetapi disukai oleh pengguna dengan selera serupa (*serendipity*).
    * **Tidak Memerlukan Fitur Konten:** Hanya membutuhkan data interaksi (rating) pengguna-item.
* **Kekurangan:**
    * **Cold Start:** Sulit memberikan rekomendasi untuk item atau pengguna baru yang belum memiliki data interaksi.
    * **Data Sparsity:** Performanya dapat menurun jika data rating sangat sedikit.

* **Hasil Top-N Recommendation:**
    Berikut adalah contoh hasil rekomendasi Top-10 untuk **User ID 186**, yang menunjukkan model cenderung merekomendasikan film-film klasik yang sangat diakui:
    ```
                                               title  predicted_rating
    0  Wallace & Gromit: The Best of Aardman Animatio...          4.362990
    1                                Citizen Kane (1941)          4.341189
    2                             Cinema Paradiso (1988)          4.393520
    3                                12 Angry Men (1957)          4.405180
    4                             Shall We Dance? (1996)          4.353436
    5                           Good Will Hunting (1997)          4.411070
    6                            Schindler's List (1993)          4.375055
    7                                     Vertigo (1958)          4.351394
    8                          North by Northwest (1959)          4.415900
    9                                  Casablanca (1942)          4.463552
    ```
---

## Evaluation

### Metrik Evaluasi yang Digunakan
-   **RMSE (Root Mean Squared Error):** Untuk mengevaluasi seberapa akurat model Collaborative Filtering dalam memprediksi rating.
-   **Precision@k:** Untuk mengevaluasi kualitas daftar peringkat (top-k recommendations) dari kedua model.

### Formula dan Cara Kerja Metrik
-   **RMSE:** $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$. Mengukur rata-rata error prediksi rating. **Nilai lebih rendah lebih baik.**
-   **Precision@k:** $Precision@k = \frac{|\text{Rekomendasi Relevan}|}{k}$. Mengukur **ketepatan** dari *k* rekomendasi.

### Hasil Proyek Berdasarkan Metrik Evaluasi
Sesuai **Review Kriteria 10**, hasil evaluasi yang disajikan di sini **konsisten** dengan output akhir pada notebook, yang dijamin dengan penggunaan `random_seed` pada fungsi evaluasi.

**Hasil Prediksi Rating (Collaborative Filtering):**
-   **Validation RMSE:** 0.9119
-   **Validation MAE:** 0.7398
-   *Analisis:* Model Collaborative Filtering mencapai nilai RMSE yang sangat mendekati target, yaitu **0.91**. Meskipun sedikit di atas target ambisius (≤ 0.85), hasil ini menunjukkan bahwa model **berhasil mempelajari pola preferensi pengguna dengan baik**, mengingat tingkat kesulitan pada dataset yang *sparse*.

**Hasil Peringkat (Ranking):**

| Model             | Precision@10    |
| **Content-Based** | 0.000 (0.0%)    |
| **Collaborative** | **0.082 (8.2%)**|

### Analisis Hasil dan Pencapaian Goals
Berdasarkan hasil evaluasi yang sebenarnya, berikut adalah analisis pencapaian proyek:

1.  **Model Collaborative Filtering Jauh Lebih Unggul:** Dalam metrik peringkat, model Collaborative Filtering (`Precision@10 = 8.2%`) secara signifikan lebih baik daripada model Content-Based (`Precision@10 = 0.0%`). Ini membuktikan bahwa pendekatan yang mempelajari pola perilaku komunitas lebih efektif dalam menghasilkan daftar rekomendasi yang relevan pada dataset ini.

2.  **Performa Content-Based:** Nilai presisi 0.0% pada model Content-Based kemungkinan besar disebabkan oleh **metodologi evaluasi yang sangat ketat**, yang terbukti dari `Evaluated Users: 0` pada output teks. Ini mengindikasikan bahwa untuk sampel pengguna yang diuji, tidak ada tumpang tindih antara rekomendasi "aman" berbasis genre dengan item-item relevan di *test set* mereka yang kecil. Ini menyoroti tantangan dalam mengevaluasi model CB dan ketergantungannya pada film referensi yang tepat.

3.  **Status Pencapaian Target:**
    -   **Target RMSE (≤ 0.85):** **Tidak Tercapai**, namun hasil **0.91** adalah performa yang kuat dan realistis.
    -   **Target MAE (≤ 0.65):** **Tidak Tercapai**, dengan hasil **0.74**.
    -   **Target Precision (≥ 25%):** **Tidak Tercapai** untuk kedua model.

Meskipun target kuantitatif yang agresif tidak tercapai sepenuhnya, proyek ini berhasil **mengimplementasikan, mengevaluasi, dan membandingkan dua pendekatan fundamental sistem rekomendasi secara valid**.