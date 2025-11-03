# transformer_translation
Tugas Eksplorasi Transformer ( 122140127,  122140165,  122140145 )
ini adalah rangkuman lengkap dari keseluruhan alur kerja proyek *Machine Translation* (Penerjemah Mesin) dari Bahasa Inggris ke Bahasa Prancis menggunakan arsitektur Transformer.

Proyek ini dibagi menjadi empat tahap utama yang saling berurutan:

### 1. Persiapan Data (Bagian 1)

Tahap ini adalah fondasi dari segalanya. Tujuannya adalah mengambil data teks mentah dan mengubahnya menjadi format yang siap "dimakan" oleh model *deep learning*.

* **Pemuatan Data**: Data teks paralel (Inggris dan Prancis) dimuat dari Google Drive.
* **Tokenisasi**: **Spacy** digunakan untuk membuat "tokenizer" (pemecah kata) untuk kedua bahasa (`tokenize_en` dan `tokenize_fr`).
* **Pemisahan Data**: Dataset dibagi menjadi dua set: `train_df` (90% untuk melatih model) dan `val_df` (10% untuk menguji model).
* **Pembuatan Vocabulary**: Dua "kamus" (`vocab_src` dan `vocab_tgt`) dibuat dari data latih. Ini memetakan setiap kata unik ke angka (indeks) dan menambahkan token khusus seperti `<pad>` (padding) dan `<bos>`/`<eos>` (penanda awal/akhir kalimat).
* **DataLoader**: Terakhir, `Dataset` kustom dan `DataLoader` PyTorch dibuat. `DataLoader` secara efisien mengelompokkan data ke dalam *batch* dan menggunakan `collate_fn` untuk **melakukan padding** (menyamakan panjang kalimat dalam satu *batch* dengan token `<pad>`).

### 2. Definisi Arsitektur (Bagian 2)

Setelah data siap, tahap ini mendefinisikan "cetak biru" (arsitektur) dari model **Transformer** itu sendiri.

* **Komponen Pendukung**: Dua kelas penting dibuat: `TokenEmbedding` (untuk mengubah angka/indeks kata menjadi vektor makna) dan `PositionalEncoding` (untuk memberi model informasi tentang *urutan* kata).
* **Model Utama**: Kelas `Seq2SeqTransformer` menggabungkan semuanya. Ini membungkus `nn.Transformer` inti dari PyTorch (yang berisi tumpukan Encoder dan Decoder) dan menambahkan layer `generator` di akhir untuk memprediksi kata keluaran.
* **Masking**: Fungsi krusial `create_mask` dibuat. Ini menghasilkan dua jenis topeng: **padding mask** (agar model mengabaikan token `<pad>`) dan **subsequent mask** (agar *decoder* tidak "curang" melihat kata-kata di masa depan saat latihan).

### 3. Proses Pelatihan (Bagian 3)

Di sinilah "pembelajaran" terjadi. Model dari Bagian 2 dilatih menggunakan data dari Bagian 1.

* **Inisialisasi**: Model diinisialisasi (dengan hyperparameter yang lebih ringan untuk **mencegah error memori/OOM**). *Loss function* (`CrossEntropyLoss` yang mengabaikan *padding*) dan *optimizer* (`Adam`) juga didefinisikan.
* **Loop Pelatihan**: Model dijalankan selama **1 epoch**. Untuk setiap *batch* data latih, model membuat prediksi, menghitung *error* (*loss*), dan menggunakan *optimizer* untuk memperbarui bobotnya (proses `loss.backward()` dan `optimizer.step()`).
* **Evaluasi**: Setelah epoch selesai, fungsi `evaluate` dijalankan pada data validasi. Ini sangat penting untuk mengukur kinerja model pada data yang belum pernah dilihatnya, yang dilaporkan sebagai **ValLoss** (rata-rata *error*) dan **ValAcc** (akurasi prediksi kata).

### 4. Inferensi / Penerapan (Bagian 4)

Ini adalah tahap akhir di mana model yang sudah dilatih (dari Bagian 3) digunakan untuk melakukan tugasnya.

* **Logika Dekode**: Fungsi `greedy_decode` diimplementasikan. Ini adalah inti dari penerjemah. Ia mengambil kalimat Inggris, menjalankannya melalui *encoder* satu kali, lalu menggunakan *decoder* berulang kali untuk menghasilkan terjemahan kata demi kata, selalu memilih kata dengan probabilitas tertinggi.
* **Fungsi Wrapper**: Fungsi `translate` dibuat sebagai pembungkus yang ramah pengguna. Ia menangani seluruh proses: mengambil kalimat teks, mengubahnya jadi tensor, memanggil `greedy_decode`, dan mengubah tensor hasil kembali menjadi teks yang bisa dibaca.
* **Eksekusi**: Akhirnya, model diuji dengan menerjemahkan beberapa contoh kalimat (dari data validasi dan input kustom) untuk menunjukkan bahwa keseluruhan alur kerja dari awal hingga akhir berhasil.
