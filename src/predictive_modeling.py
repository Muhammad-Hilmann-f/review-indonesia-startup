import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

def predictive_modeling(df):
    """
    Membuat model prediksi sentimen dan kategori rating berdasarkan review pengguna.
    """
    print("\n🤖 PREDICTIVE MODELING")
    print("-" * 37)

    # Prepare features
    if 'review_text_clean' not in df.columns or df['review_text_clean'].isna().all():
        print("❌ Text data tidak tersedia untuk modeling")
        return None

    # Text vectorization
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    X_text = vectorizer.fit_transform(df['review_text_clean'].fillna(''))

    # Additional features
    additional_features = []
    if 'review_length' in df.columns:
        additional_features.append('review_length')
    if 'word_count' in df.columns:
        additional_features.append('word_count')

    if additional_features:
        X_additional = df[additional_features].fillna(0)
        X_additional_scaled = StandardScaler().fit_transform(X_additional)
        # Combine text and additional features
        X = hstack([X_text, X_additional_scaled])
    else:
        X = X_text.toarray()

    # Model 1: Sentiment Prediction
    print("\n1. Training Sentiment Prediction Model...")
    y_sentiment = df['sentiment_label'].fillna('Neutral')

    X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'y_test': y_test
        }
        print(f"  {name} Accuracy: {accuracy:.3f}")

    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']

    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")

    # Model 2: Rating Category Prediction
    print("\n2. Training Rating Category Prediction Model...")
    y_rating = df['rating_category'].fillna('Neutral')

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rating, test_size=0.2, random_state=42)

    rating_model = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rating_model.fit(X_train_r, y_train_r)
    y_pred_r = rating_model.predict(X_test_r)
    rating_accuracy = accuracy_score(y_test_r, y_pred_r)

    print(f"  Rating Category Model Accuracy: {rating_accuracy:.3f}")

    # Visualize model performance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    bars = ax1.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)

    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    # 2. Confusion Matrix for best sentiment model
    cm_sentiment = confusion_matrix(results[best_model_name]['y_test'], results[best_model_name]['predictions'])
    sns.heatmap(cm_sentiment, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    # 3. Confusion Matrix for rating model
    cm_rating = confusion_matrix(y_test_r, y_pred_r)
    sns.heatmap(cm_rating, annot=True, fmt='d', cmap='Greens', ax=ax3)
    ax3.set_title('Confusion Matrix - Rating Category', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')

    # 4. Feature importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        if hasattr(rf_model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(len(rf_model.feature_importances_))]
            importances = rf_model.feature_importances_
            top_indices = np.argsort(importances)[-10:]

            ax4.barh(range(10), importances[top_indices], color='orange')
            ax4.set_yticks(range(10))
            ax4.set_yticklabels([f'Feature {i}' for i in top_indices])
            ax4.set_title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig('output/predictive_modeling.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed classification report
    print(f"\n📊 Detailed Classification Report - {best_model_name}:")
    print(classification_report(results[best_model_name]['y_test'], results[best_model_name]['predictions']))

    # Save models
    return {
        'sentiment_model': best_model,
        'rating_model': rating_model,
        'vectorizer': vectorizer,
        'scaler': StandardScaler().fit(df[additional_features].fillna(0)) if additional_features else None
    }