# 🚀 Startup Review Analyzer  

**Analisis komprehensif terhadap review aplikasi startup Indonesia menggunakan Data Science dan Machine Learning.**  

## 📋 Deskripsi  
Startup Review Analyzer adalah tool analisis data yang dirancang khusus untuk menganalisis review aplikasi startup Indonesia seperti **Gojek, Tokopedia, Dana, OVO, Traveloka, Ruangguru, Bukalapak, dan Blibli**.  
Tool ini menggunakan **Natural Language Processing (NLP)**, **sentiment analysis**, dan **machine learning** untuk memberikan **insights bisnis yang actionable**.  

---

## ✨ Fitur Utama  

### 🔍 **Analisis Data Komprehensif**  
✔ **Data Integration** → Menggabungkan dan membersihkan data dari multiple sources  
✔ **Exploratory Data Analysis (EDA)** → Visualisasi distribusi rating, trend temporal, dan performa per sektor  
✔ **Data Quality** → Handling missing values, duplicate removal, dan standardisasi format data  

### 🎭 **Sentiment Analysis**  
✔ **TextBlob Integration** → Analisis sentimen otomatis dari review text  
✔ **Sentiment Classification** → Kategorisasi review menjadi **Positive, Negative, dan Neutral**  
✔ **Sentiment Scoring** → Scoring numerik untuk analisis kuantitatif  
✔ **Sentiment Visualization** → Word clouds dan distribusi sentimen per aplikasi  

### 👥 **Customer Segmentation**  
✔ **K-Means Clustering** → Segmentasi pelanggan berdasarkan behavior patterns  
✔ **User Profiling** → Analisis karakteristik reviewer berdasarkan rating dan sentiment  
✔ **Segment Analysis** → Identifikasi customer segments untuk targeted strategies  

### 🤖 **Predictive Modeling**  
✔ **Multiple Algorithms** → Logistic Regression, Decision Tree, Random Forest  
✔ **TF-IDF Vectorization** → Text feature extraction untuk machine learning  
✔ **Model Comparison** → Evaluasi performa model dengan accuracy metrics  
✔ **Prediction Capabilities** → Prediksi sentiment dan rating category untuk review baru  

### 💡 **Business Intelligence**  
✔ **Performance Metrics** → KPI tracking untuk setiap aplikasi dan sektor  
✔ **Competitive Analysis** → Benchmark performa antar aplikasi startup  
✔ **Issue Detection** → Identifikasi common issues dari negative reviews  
✔ **Actionable Recommendations** → Rekomendasi bisnis berdasarkan data insights  

### 📊 **Interactive Dashboard**  
✔ **Comprehensive Visualization** → Dashboard lengkap dengan **12 key metrics**  
✔ **Export Capabilities** → Export hasil dalam format **CSV dan PNG**  
✔ **Real-time Insights** → Summary statistics dan trend analysis  

---

## 🛠️ **Teknologi yang Digunakan**  

### 📂 **Data Processing & Analysis**  
- `pandas` → Data manipulation and analysis  
- `numpy` → Numerical computing  
- `matplotlib` → Static plotting  
- `seaborn` → Statistical visualization  
- `plotly` → Interactive plotting  

### 📑 **Natural Language Processing (NLP)**  
- `nltk` → Natural Language Toolkit  
- `textblob` → Sentiment analysis  
- `wordcloud` → Word cloud generation  
- `scikit-learn` → TF-IDF vectorization  

### 🧠 **Machine Learning**  
- `scikit-learn` → ML algorithms and metrics  
  - `LogisticRegression`  
  - `DecisionTreeClassifier`  
  - `RandomForestClassifier`  
  - `KMeans clustering`  
  - `TfidfVectorizer`  
  - `train_test_split`  

### 📊 **Visualization & Reporting**  
- `matplotlib.pyplot` → Advanced plotting  
- `seaborn` → Statistical graphics  
- `plotly.express` → Interactive visualizations  

---

```## 📁 **Struktur Folder & Output File**  
📦 src/
┣ 📂 data_loader.py → Load & combine datasets
┣ 📂 preprocessing.py → Data cleaning & feature engineering
┣ 📂 eda.py → Exploratory Data Analysis
┣ 📂 sentiment_analysis.py → NLP Sentiment Modeling
┣ 📂 customer_segmentation.py → K-Means Clustering
┣ 📂 predictive_modeling.py → Machine Learning Model
┣ 📂 insights.py → Business recommendations
┣ 📂 dashboard.py → Visualisasi & summary
┗ 📂 config.py → Pengaturan globa
📦 output/  
┣ 📜 processed_startup_reviews.zip → Dataset hasil cleaning dalam format ZIP
┣ 📜 analysis_summary.csv → Statistik utama
┣ 📜 app_performance_metrics.csv → Performansi aplikasi berdasarkan rating & sentimen
┣ 📜 sector_analysis.csv → Analisis performa sektor startup
┣ 📜 business_recommendations.json → Rekomendasi bisnis
┣ 📊 eda_analysis.png → Visualisasi eksplorasi data
┣ 📊 sentiment_analysis.png → Grafik sentimen pengguna
┣ 📊 customer_segmentation.png → Cluster pengguna
┣ 📊 predictive_modeling.png → Hasil model prediksi
┗ 📊 dashboard_summary.png → Dashboard visualisas
```
---
## 🔧 **Cara Install & Menjalankan**  
1️⃣ **Clone Repository:**  
```bash
git clone https://github.com/Muhammad-Hilmann-f/review-indonesia-startup.git
cd review-indonesia-startup
pip install -r requirements.txt
python main.py
```
## 📊 Hasil Analisis & Visualisasi
---
**✅ Total Reviews Setelah Cleaning: 529,760**
---
**✅ Rata-rata Rating: 3.36**
---
**✅ Rentang Waktu Data: 2014 - 2024**
---
---
**✅ Aplikasi dengan Review Terbanyak:**
  - Traveloka → 145,890 reviews
  - Gojek → 116,016 reviews
  - Dana → 106,465 reviews
    
    ---
**✅ Distribusi Sentimen:**
  - Neutral: 467,466 reviews
  - Positive: 57,651 reviews
  - Negative: 4,643 reviews
    
    ---
**✅ Aplikasi dengan Sentimen Terbaik:**
  - Traveloka → Sentiment Score: 0.105
  - Dana → Sentiment Score: 0.039
  - Bukalapak → Sentiment Score: 0.034

    ---
**✅ Customer Segmentation (K-Means Clustering):**
  - Cluster 0 → Rata-rata rating: 2.80 (Review panjang, rating rendah)
  - Cluster 1 → Rata-rata rating: 4.27 (Review pendek, rating tinggi)
  - Cluster 2 → Rata-rata rating: 3.85 (Sedang)

  ---
**✅ Predictive Modeling:**
  - Model yang diuji: Logistic Regression, Decision Tree, Random Forest
  - Best Model untuk Sentimen: Decision Tree (Accuracy: 94.4%)
  - Model Rating Category Accuracy: 78.2%
  - Confusion Matrix & Feature Importance divisualisasikan

    ---
**✅ Dashboard Summary:**
  - Visualisasi lengkap dalam satu tampilan
  - Disimpan di output/dashboard_summary.png

  ---
**✅ Performa sektor startup:**
- ⭐ Travel terbaik → Rating 4.27 & Sentiment Score 0.105
- ❌ Fintech terburuk → Rating 2.38 & Sentiment Score 0.039

  ---
**✅ Rekomendasi bisnis utama:**
- 🎯 Fokus pada aplikasi dengan rating rendah untuk perbaikan layanan
- ⭐ Pelajari strategi aplikasi dengan rating tinggi untuk meningkatkan performa
- ⚠️ Perhatikan sentimen negatif yang tinggi, segera tingkatkan customer support
- 📈 Monitor tren sentimen dan rating untuk mengidentifikasi pola perubahan

---
