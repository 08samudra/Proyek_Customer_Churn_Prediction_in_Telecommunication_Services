# ML Terapan Proyek 1 | Customer Churn Prediction in Telecommunication Services

###### Disusun oleh : Yoga Samudra

Ini adalah proyek pertama analisis prediktif untuk memenuhi submission Dicoding Kelas Machine Learning Terapan.

Proyek ini membangun model *machine learning* yang dapat memprediksi churn pelanggan pada layanan telekomunikasi.

## Domain Proyek

### Latar Belakang

Domain proyek *Customer Churn Prediction in Telecommunication Services* berada pada industri layanan telekomunikasi, sebuah sektor yang sangat kompetitif dan terus berkembang. Perusahaan telekomunikasi menyediakan berbagai layanan komunikasi seperti internet, telepon, dan televisi berlangganan kepada jutaan pelanggan.

Dalam domain ini, mempertahankan pelanggan lama menjadi tantangan utama karena biaya akuisisi pelanggan baru jauh lebih tinggi dibandingkan dengan mempertahankan pelanggan yang sudah ada. Salah satu indikator penting dalam manajemen pelanggan adalah churn, yaitu kondisi ketika pelanggan berhenti menggunakan layanan dan berpindah ke kompetitor.

Relevansi proyek ini dengan industri telekomunikasi adalah sebagai berikut:

1. Prediksi Churn Secara Akurat: Model prediktif yang dikembangkan dalam proyek ini bertujuan untuk membantu perusahaan telekomunikasi mengidentifikasi pelanggan yang berisiko tinggi untuk churn. Dengan menganalisis data historis pelanggan seperti masa berlangganan, jenis layanan yang digunakan, pola pembayaran, dan keluhan layanan, model ini dapat memprediksi churn secara lebih akurat.

2. Peningkatan Loyalitas dan Retensi Pelanggan: Dengan mengetahui pelanggan yang berpotensi churn, perusahaan dapat melakukan tindakan preventif seperti penawaran khusus, peningkatan layanan, atau program loyalitas. Hal ini memungkinkan peningkatan retensi pelanggan dan mencegah kehilangan pendapatan jangka panjang.

Manfaat bagi Perusahaan Telekomunikasi:
- Efisiensi Strategi Retensi: Dengan memanfaatkan prediksi churn, perusahaan dapat mengalokasikan sumber daya secara lebih efektif untuk mempertahankan pelanggan bernilai tinggi, sehingga meningkatkan efisiensi program loyalitas.
- Pengambilan Keputusan Berbasis Data: Keputusan bisnis dalam hal intervensi pelanggan menjadi lebih terarah karena didasarkan pada analisis dan skor risiko churn yang dihasilkan oleh model.

Manfaat bagi Pelanggan:
- Layanan yang Lebih Personal dan Relevan: Pelanggan yang berisiko churn akan menerima pendekatan layanan yang lebih sesuai dengan kebutuhan mereka, misalnya berupa paket khusus, peningkatan layanan, atau insentif loyalitas.
- Pengalaman Pelanggan yang Lebih Baik: Dengan adanya prediksi dan intervensi dini, pelanggan akan merasa lebih dihargai dan dipedulikan, yang dapat meningkatkan kepuasan dan loyalitas jangka panjang.

Secara keseluruhan, model analisis prediktif dalam proyek *Customer Churn Prediction in Telecommunication Services* memberikan manfaat strategis bagi perusahaan dan pelanggan. Dengan kemampuan prediksi yang akurat, perusahaan dapat mengurangi tingkat churn, mempertahankan pendapatan, serta memberikan pengalaman pelanggan yang lebih baik dan lebih personal.

## Business Understanding

### **Problem Statements**

Dalam industri telekomunikasi, kehilangan pelanggan (customer churn) berdampak langsung terhadap pendapatan dan keberlanjutan bisnis. Oleh karena itu, memahami dan memprediksi perilaku churn sangat penting. Beberapa pernyataan masalah yang diangkat dalam proyek ini adalah:

1. **Bagaimana cara mengidentifikasi pelanggan yang berpotensi untuk berhenti berlangganan layanan telekomunikasi?**
2. **Faktor apa saja yang paling memengaruhi keputusan pelanggan untuk berhenti menggunakan layanan?**
3. **Bagaimana perusahaan dapat menggunakan hasil prediksi churn untuk merancang strategi retensi pelanggan yang lebih efektif?**

### **Goals**

Berdasarkan pernyataan masalah di atas, proyek ini bertujuan untuk:

1. **Mengembangkan model machine learning yang mampu memprediksi churn pelanggan dengan akurasi tinggi.**
2. **Mengidentifikasi fitur-fitur penting yang memengaruhi churn, seperti jenis kontrak, penggunaan layanan, dan metode pembayaran.**
3. **Memberikan rekomendasi berbasis data yang dapat digunakan oleh perusahaan untuk merancang intervensi preventif terhadap pelanggan yang berisiko churn.**

### **Solution Statements**

Untuk mencapai tujuan di atas, solusi yang dirancang dalam proyek ini mencakup:

* **Penggunaan dua model machine learning:**

  * **Logistic Regression** sebagai baseline model karena interpretabilitas dan kesederhanaannya.
  * **Random Forest Classifier** untuk menangkap hubungan non-linear dan interaksi antar fitur, serta memberikan feature importance.

* **Evaluasi model menggunakan metrik klasifikasi** seperti:

  * Akurasi
  * Precision
  * Recall
  * F1-Score

* **Peningkatan performa model melalui proses tuning hyperparameter** pada model Random Forest untuk mencapai hasil yang optimal.

Dengan pendekatan ini, solusi prediktif yang dikembangkan dapat memberikan wawasan yang jelas bagi manajemen dalam mengambil keputusan retensi pelanggan berdasarkan probabilitas churn.

Terima kasih, dan pertanyaanmu sangat tepat! Mari kita review terlebih dahulu.

## **Data Understanding**

Tahap Data Understanding bertujuan untuk memahami karakteristik data pelanggan Telco sebelum membangun model prediksi churn. Dataset berisi 7043 entri pelanggan dengan 21 fitur, yang terdiri dari data demografis, layanan yang digunakan, dan informasi pembayaran.

### 1. **Tipe Data**

* Fitur terdiri dari:

  * **Fitur kategorikal** seperti: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `InternetService`, dll.
  * **Fitur numerik**: `tenure`, `MonthlyCharges`, `TotalCharges`.

### 2. **Pemeriksaan Nilai Hilang**

* Fitur `TotalCharges` memiliki nilai kosong dalam bentuk string spasi `" "`.

  * Setelah konversi ke numerik, ditemukan 11 nilai kosong.
  * Nilai-nilai ini dihapus karena proporsinya sangat kecil (\~0.15%).

### 3. **Deskripsi Statistik**

**Fitur Numerik (Setelah dibersihkan):**

| Fitur          | Mean    | Median  | Std     | Min   | Max     |
| -------------- | ------- | ------- | ------- | ----- | ------- |
| tenure         | 32.42   | 29      | 24.55   | 1     | 72      |
| MonthlyCharges | 64.79   | 70.35   | 30.08   | 18.25 | 118.75  |
| TotalCharges   | 2283.30 | 1397.48 | 2266.77 | 18.80 | 8684.80 |

* Pelanggan yang churn cenderung memiliki `tenure` yang lebih pendek dan `TotalCharges` yang lebih rendah dibandingkan pelanggan yang tidak churn.

### 4. **Distribusi Fitur Kategorikal**

Distribusi fitur menunjukkan bahwa mayoritas pelanggan:

* Memiliki layanan telepon (PhoneService): 90.3%
* Tidak memiliki tanggungan (Dependents): 70.1%
* Berlangganan kontrak bulanan (Month-to-month): 55.1%
* Menggunakan metode pembayaran Electronic check: 33.6%
* Memiliki koneksi internet: 78.4%

**Churn berdasarkan fitur:**

* Pelanggan tanpa pasangan atau tanggungan cenderung lebih tinggi tingkat churn-nya.
* Kontrak jangka panjang (1–2 tahun) memiliki churn yang jauh lebih rendah dibandingkan kontrak bulanan.

**Hasil visualisasi:**

> ![Distribusi Churn berdasarkan Tipe Kontrak](assets/images/distribusi_tipe_kontrak.png)

### 5. **Korelasi Antar Fitur Numerik**

Korelasi menunjukkan hubungan yang cukup kuat antara:

* `tenure` dan `TotalCharges`: **0.83**
* `MonthlyCharges` dan `TotalCharges`: **0.65**
* `tenure` dan `MonthlyCharges`: **0.25**

**Hasil visualisasi:**

> Heatmap Korelasi Numerik
> ![Heatmap Korelasi](assets/images/kolerasi_antar_fitur_numerik.png)

### 6. **Korelasi dengan Target (Churn)**

Korelasi terhadap variabel `Churn` (dalam bentuk numerik: 0 = No, 1 = Yes):

| Fitur          | Korelasi  |
| -------------- | --------- |
| tenure         | **-0.35** |
| MonthlyCharges | +0.19     |
| TotalCharges   | -0.20     |

* **Tenure memiliki korelasi negatif yang cukup kuat**: pelanggan dengan masa langganan lebih lama cenderung tidak churn.
* **MonthlyCharges memiliki korelasi positif**: pelanggan dengan biaya bulanan lebih tinggi cenderung churn.

### 7. **Kesimpulan Sementara**

* **Pelanggan churn biasanya baru bergabung dan membayar biaya bulanan tinggi.**
* **Jenis kontrak sangat memengaruhi churn.**
* **Korelasi numerik mendukung bahwa `tenure` dan `TotalCharges` adalah indikator penting.**

## Data Preparation

Pada tahap ini, dilakukan sejumlah proses persiapan data agar data mentah yang tersedia dapat digunakan secara efektif oleh model klasifikasi. Berikut adalah teknik dan tahapan data preparation yang telah dilakukan:

### 1. Pembersihan Nilai Kosong  
Dataset asli memiliki kolom `TotalCharges` yang seharusnya bertipe numerik, namun terdeteksi memiliki nilai kosong yang tersimpan sebagai string kosong (`''`).  
Untuk mengatasi hal ini:
- Kolom `TotalCharges` dikonversi menjadi tipe numerik menggunakan `pd.to_numeric()` dengan parameter `errors='coerce'`.
- Baris dengan nilai kosong setelah konversi dihapus dari dataset menggunakan `dropna()`.

Tujuannya adalah memastikan tidak ada noise data yang mengganggu saat proses pelatihan model.

### 2. Konversi Target Variabel  
Fitur target `Churn` yang berisi nilai kategorikal `'Yes'` dan `'No'` dikonversi ke nilai numerik (`1` dan `0`) dengan menggunakan `.map()`:

```python
df['Churn_numerik'] = df['Churn'].map({'No': 0, 'Yes': 1})
````

Ini diperlukan agar algoritma klasifikasi dapat memproses target sebagai variabel numerik biner.

### 3. Encoding Fitur Kategorikal

Fitur-fitur kategorikal seperti `gender`, `Partner`, `Dependents`, `InternetService`, `Contract`, dan `PaymentMethod` tidak dapat digunakan langsung dalam pemodelan karena bersifat non-numerik.
Untuk itu, dilakukan proses **One Hot Encoding** menggunakan fungsi `pd.get_dummies()`:

```python
df_encoded = pd.get_dummies(df.drop(columns=['customerID', 'Churn']), drop_first=True)
```

* Fitur `customerID` dihapus karena merupakan ID unik yang tidak relevan.
* Parameter `drop_first=True` digunakan untuk menghindari dummy variable trap.

### 4. Seleksi Fitur

Dari hasil eksplorasi data sebelumnya (Langkah 11–13), diketahui bahwa fitur-fitur numerik `tenure`, `MonthlyCharges`, dan `TotalCharges` memiliki hubungan dengan `Churn`, terutama `tenure` yang memiliki korelasi negatif cukup kuat.
Fitur-fitur ini dipertahankan karena dianggap relevan dalam prediksi churn pelanggan.

### 5. Normalisasi Fitur Numerik

Untuk memastikan semua fitur numerik berada dalam skala yang seragam (terutama untuk model Logistic Regression), dilakukan proses **normalisasi** menggunakan `MinMaxScaler`:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
```

### 6. Pembagian Data Training dan Testing

Data dibagi menjadi data pelatihan (training set) dan pengujian (testing set) dengan rasio **80:20** menggunakan fungsi `train_test_split`:

```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns='Churn_numerik')
y = df_encoded['Churn_numerik']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

* Stratifikasi dilakukan berdasarkan target `Churn_numerik` untuk menjaga proporsi kelas seimbang antara data training dan testing.

---

Dengan seluruh tahapan ini, dataset telah siap digunakan untuk pelatihan model klasifikasi seperti **Logistic Regression**, **Random Forest**, dan **XGBoost**.
