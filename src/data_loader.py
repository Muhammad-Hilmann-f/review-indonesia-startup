import pandas as pd
import os

def load_and_combine_data(folder_path):
    """
    Menggabungkan semua file CSV reviews startup dari folder dataset.
    """
    print("\n📂 LOADING & COMBINING DATA")
    print("-" * 40)

    # Daftar file CSV
    csv_files = [
        'bukalapak.csv', 'dana.csv', 'gojek.csv', 'ovo.csv',
        'tokopedia.csv', 'traveloka.csv', 'ruangguru.csv', 'blibli.csv'
    ]

    combined_data = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        try:
            df = pd.read_csv(file_path)

            # Standardisasi nama kolom
            df.columns = df.columns.str.lower().str.replace(' ', '_')

            # Tambahkan identifikasi aplikasi
            app_name = file.replace('.csv', '')
            df['app_name'] = app_name

            # Tambahkan kategori sektor
            sectors = {
                'gojek': 'Transportation', 'dana': 'Fintech', 'ovo': 'Fintech',
                'tokopedia': 'E-commerce', 'bukalapak': 'E-commerce', 'blibli': 'E-commerce',
                'traveloka': 'Travel', 'ruangguru': 'Education'
            }
            df['sector'] = sectors.get(app_name, 'Other')

            combined_data.append(df)
            print(f"✓ {file}: {len(df)} rows loaded")

        except FileNotFoundError:
            print(f"✗ {file} tidak ditemukan")
        except Exception as e:
            print(f"✗ Error loading {file}: {str(e)}")

    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)

        # Standardisasi nama kolom umum
        column_mapping = {
            'content': 'review_text',
            'score': 'rating',
            'created_at': 'review_date'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in combined_df.columns:
                combined_df.rename(columns={old_col: new_col}, inplace=True)

        print(f"\n🎉 Total data tergabung: {len(combined_df)} rows")
        print(f"📊 Shape: {combined_df.shape}")
        print(f"🏢 Apps: {list(combined_df['app_name'].unique())}")

        return combined_df
    else:
        print("❌ Tidak ada data yang berhasil digabungkan")
        return None