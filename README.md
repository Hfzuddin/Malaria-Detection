# Malaria Detection System

Sistem ini adalah sebuah aplikasi berasaskan web untuk mengesan parasit Malaria di dalam imej sel darah (Blood Smear Scans) menggunakan model *Convolutional Neural Network (CNN)* dan *Support Vector Machine (SVM)*.

Aplikasi ini menggunakan:
- **Backend:** Python & Flask
- **Frontend:** React, TailwindCSS (menggunakan Babel Standalone & CDN)
- **AI Model:** PyTorch (EfficientNet)

## Prasyarat
Sebelum menggunakan sistem ini, pastikan komputer anda telah dipasang dengan:
1. **Python 3.8** atau lebih baharu.
2. `pip` (Python package installer).

## Cara Pemasangan (Installation)

1. **Muat turun kod sumber (Clone repository)**
   Buka terminal/Command Prompt dan jalankan:
   ```bash
   git clone https://github.com/UsernameAnda/Malaria-Detection-System.git
   cd "Malaria-Detection-System"
   ```

2. **Pasang perpustakaan (Dependencies)**
   Adalah sangat digalakkan untuk menggunakan *Virtual Environment*. Walau bagaimanapun, anda boleh terus memasang *requirements* dengan menaip:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan Sistem (Run the Application)

1. Masuk ke dalam folder `src`:
   ```bash
   cd src
   ```

2. Jalankan pelayan (server) Flask:
   ```bash
   python app.py
   ```

3. Buka pelayar web (browser) anda dan pergi ke pautan berikut:
   **http://127.0.0.1:5500**

## Cara Penggunaan
1. Di halaman **Dashboard**, klik **Upload Image** atau **Upload Folder** untuk memuat naik imej sel darah (format JPG/PNG).
2. Tekan butang **Analyze Images**. Sistem akan mengambil masa beberapa saat untuk memproses imej menggunakan model AI.
3. Setelah selesai, anda akan dibawa ke halaman **Scanned Results** untuk melihat sama ada sel tersebut dijangkiti (Infected) atau bebas parasit (Uninfected).
4. Keputusan lalu akan sentiasa disimpan secara automatik di dalam halaman ini.
