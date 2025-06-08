#  Konfigurasi Global Proyek

# Path ke dataset
DATASET_PATH = r"F:\Tugas-Kuliah\data sience\dataset"

# Folder output untuk menyimpan hasil analisis
OUTPUT_FOLDER = "output/"

# Hyperparameter untuk model prediksi
RANDOM_FOREST_PARAMS = {
    'n_estimators': 50,
    'max_depth': 12,
    'min_samples_split': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Konfigurasi visualisasi
PLOT_STYLE = "default"

# Variabel lain yang sering dipakai untuk text preprocessing
TEXT_CLEANING_CONFIG = {
    "remove_special_chars": True,
    "lowercase": True
}