# Kalkulator Matriks — Web App (Streamlit)

**Deskripsi singkat**  
Aplikasi web berbasis **Streamlit** untuk operasi matriks numerik: nilai eigen, vektor eigen, determinan, invers, solusi sistem linear, dan dekomposisi (LU, Cholesky, Doolittle, Crout).

---

## Fitur
* Input matriks melalui textarea atau unggah file CSV.  
* Hitung nilai eigen dan vektor eigen.  
* Dekomposisi: LU, Cholesky (jika positif-definit), Doolittle, Crout.  
* Operasi dasar: determinan, invers, rank, transpose.  
* Penyelesaian sistem linear Ax = b.  
* Tampilkan hasil numerik dan (opsional) langkah singkat perhitungan.

---

## Prasyarat
* Python 3.8+  
* Paket Python: `streamlit`, `numpy`, `scipy`, `sympy` (opsional untuk output simbolik).

Contoh menginstal dependensi:
```bash
pip install streamlit numpy scipy sympy
```

---

## Cara Jalankan (lokal)
1. Pastikan semua dependensi terpasang.  
2. Jalankan perintah:
```bash
streamlit run apps.py
```
3. Buka browser ke alamat yang ditampilkan (biasanya `http://localhost:8501`).

---

## Cara Input Matriks
* **Textarea**: Masukkan baris matriks tiap baris baru; pisahkan elemen dengan spasi atau koma.  
  Contoh 3x3:
  ```
  1 2 3
  4 5 6
  7 8 9
  ```
* **CSV**: Unggah file `.csv` yang berisi angka (baris → baris matriks).

Untuk sistem linear Ax = b, masukkan A (matriks koefisien) dan b (vektor kolom) dalam format yang sama.

---

## Contoh Penggunaan
* Hitung nilai eigen dan vektor eigen → pilih menu **Eigen** → masukkan matriks → tekan **Hitung**.  
* Dekomposisi LU → pilih **LU Decomposition** → masukkan matriks persegi → tekan **Compute**.  
* Selesaikan Ax = b → masukkan A dan b → tekan **Solve**.

---

## Catatan & Batasan
* Dekomposisi Cholesky memerlukan matriks simetris dan positif-definit.  
* Untuk matriks singular, invers tidak tersedia — aplikasi akan menampilkan pesan error/peringatan.  
* Hasil numerik menggunakan aritmetika floating-point; toleransi pembulatan dapat terjadi.

---

## Struktur Repo (contoh)
```
.
├─ apps.py            # Streamlit app (utamanya)
├─ requirements.txt   # daftar paket (opsional)
├─ README_streamlit.md
└─ examples/
   ├─ example1.csv
   └─ example2.csv
```

---

## Kontribusi
1. Fork repository  
2. Buat branch baru (`feature/nama-fitur`)  
3. Commit dan push perubahan  
4. Buat Pull Request

---

## Lisensi
Lisensi direkomendasikan: **MIT License**. Sesuaikan dengan kebijakan proyek Anda.

---

## Kontak
Jika butuh bantuan menyesuaikan UI, menambahkan fitur (mis. visualisasi langkah perhitungan), atau membuat distribusi Docker, beri tahu saya — saya bantu.
