import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. FUNGSIONALITAS UTAMA (Mock Data untuk pengujian)
def create_mock_data():
    """Membuat contoh data sederhana untuk pelatihan model."""
    data = {
        'text': [
            "Layanan ini sangat baik dan cepat",
            "Saya kecewa dengan kualitas produk",
            "Semua berjalan sempurna",
            "Ini adalah pengalaman terburuk",
            "Produknya melebihi harapan saya",
            "Mengapa prosesnya begitu lambat"
        ],
        'sentiment': [1, 0, 1, 0, 1, 0]  # 1: Positif, 0: Negatif
    }
    return pd.DataFrame(data)

# 2. FUNGSIONALITAS PELATIHAN MODEL
def train_sentiment_model(df):
    """Melatih model Naive Bayes sederhana."""
    
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.3, random_state=42
    )
    
    # Ekstraksi Fitur Teks
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Pelatihan Model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    return model, vectorizer, X_test, y_test

# Bagian utama yang akan dipanggil oleh Notebook
if __name__ == "__main__":
    df_data = create_mock_data()
    print("Data siap dengan kolom 'text' dan 'sentiment'")
    
    # Model dilatih, tetapi hasil tes akan dianalisis di Notebook
    model, vectorizer, X_test, y_test = train_sentiment_model(df_data)
    print("Model telah dilatih. Siap untuk dievaluasi.")