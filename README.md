# ANALISIS DAN PERBANDINGAN MODEL REGRESI UNTUK PREDIKSI NILAI TRANSAKSI E-COMMERCE MENGGUNAKAN DATASET OLIST

**UJIAN AKHIR SEMESTER MATA KULIAH BIG DATA ANALYSIS AND DATA VISUALISATION**


## Deskripsi Proyek
Proyek ini merupakan implementasi dari *pipeline* *Machine Learning* *End-to-End* untuk melakukan prediksi terhadap nilai pembayaran (*payment value*) pelanggan pada dataset E-Commerce publik asal Brazil, **Olist**.

Fokus utama dari analisis ini adalah untuk menerapkan tahapan komprehensif dalam pemrosesan *Big Data*, mulai dari persiapan data (*Data Preparation*), rekayasa fitur (*Feature Engineering*), penanganan *missing value*, hingga pemodelan komparatif menggunakan beberapa algoritma prediktif.

## Tahapan Analisis & Pemrosesan Data

### 1. Data Preparation (`preparation_dataset.py`)
Tahap awal meliputi penggabungan (*join*) data dari berbagai tabel relasional (seperti `orders`, `order_items`, `products`, `customers`, dan `payments`) untuk membentuk satu dataset utuh di level transaksi (*order level*). Data difilter hanya untuk transaksi yang statusnya telah "delivered" (berhasil dikirim).

### 2. Penanganan Missing Value & Preprocessing
Untuk menguji kehandalan model, dilakukan simulasi *missing value* dengan menghapus secara acak 5% data pada fitur-fitur numerik (`price`, `freight_value`, `product_weight_g`).
- **Imputasi Numerik:** Kolom yang kosong diisi kembali menggunakan nilai **Median**.
- **Imputasi Kategorikal:** Kolom kategorikal (`product_category_name`, `customer_state`, `payment_type`) yang kosong diisi dengan nilai modus (*mode*).

### 3. Feature Engineering & Transformasi (`main_comparison.py`)
- Pembuatan variabel baru: `total_price`, rasio pengiriman (`shipping_ratio`), dan rasio berat terhadap harga (`weight_price_ratio`).
- Transformasi target `payment_value` menggunakan fungsi logaritma natural (`log1p`) untuk menormalkan distribusi data yang cenderung *skewed*.
- Standarisasi (*Scaling*) pada data numerik menggunakan `StandardScaler`.
- *One-Hot Encoding* untuk data kategorikal menggunakan metode *dummy variables*.

## Pemodelan (Machine Learning)
Analisis dilakukan dengan membandingkan tiga algoritma Regresi pada dua skenario pemisahan data (*Train-Test Split*: 80:20 dan 70:30):

1. **XGBoost Regressor** (Model andalan yang dioptimasi)
2. **Random Forest Regressor**
3. **Linear Regression** (Sebagai *baseline model*)

Semua model dievaluasi kinerjanya dalam memprediksi *payment value* yang dikembalikan ke skala asli (*inverse transform* menggunakan `expm1`).

## Metrik Evaluasi
Performa dari setiap algoritma dikur secara detail menggunakan metrik regresi standar:
- **MAE** (*Mean Absolute Error*)
- **RMSE** (*Root Mean Squared Error*)
- **R² Score** (*Coefficient of Determination*)

## Visualisasi Data (Outputs)
Hasil eksekusi program ini akan menghasilkan sejumlah visualisasi (disimpan di folder `/outputs`), antara lain:
- `CORRELATION MATRIX.png`: Peta panas (*heatmap*) korelasi antar fitur numerik.
- `DISTRIBUSI PAYMENT VALUE.png`: Grafik distribusi target.
- `ACTUAL vs PREDICTED`: Plot persebaran nilai prediksi berbanding nilai aktual untuk memvisualisasikan akurasi model di berbagai metrik dan *split data*.
- `FEATURE IMPORTANCE`: Grafik batang yang menunjukkan atribut mana yang paling berkontribusi dalam menentukan *payment value* pada model *XGBoost* dan *Random Forest*.

## Menjalankan Program
1. Eksekusi skrip preparasi untuk menggabungkan dataset mentah (Opsional jika data *success* sudah ada):
   ```bash
   python preparation_dataset.py
   ```
2. Eksekusi program utama untuk menjalankan pipeline pemodelan dan komparasi secara penuh:
   ```bash
   python main_comparison.py
   ```

---
*Proyek ini disusun sebagai bagian dari pemenuhan tugas UAS (Ujian Akhir Semester).*
