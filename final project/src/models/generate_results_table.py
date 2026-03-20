import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================
# Load dataset
# ============================
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")
X = df["Cleaned_Abstract"]
y = df["Label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.17, random_state=42, stratify=y
)

results = []

def evaluate(model_name, model_path, vect_path, sbert=False):
    print(f"[INFO] Evaluating {model_name}...")

    if sbert:
        from sentence_transformers import SentenceTransformer
        model = joblib.load(model_path)
        vectorizer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X_test_vec = vectorizer.encode(list(X_test), show_progress_bar=False)
    else:
        model = joblib.load(model_path)
        tfidf = joblib.load(vect_path)
        X_test_vec = tfidf.transform(X_test)

    y_pred = model.predict(X_test_vec)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    accuracy = accuracy_score(y_test, y_pred)

    results.append([model_name, accuracy, precision, recall, f1])


# ============================
# Evaluate all models
# ============================
evaluate("Logistic Regression", "baseline_lr_model.pkl", "tfidf_vectorizer.pkl")
evaluate("SVM (Linear)", "svm_balanced_model.pkl", "svm_tfidf_vectorizer.pkl")
evaluate("Random Forest", "rf_model.pkl", "rf_tfidf_vectorizer.pkl")
evaluate("XGBoost", "xgb_model.pkl", "xgb_tfidf_vectorizer.pkl")
evaluate("SBERT + SVM", "sbert_svm_model.pkl", None, sbert=True)

# ============================
# Save Comparison Table
# ============================
df_results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

df_results.to_csv("model_comparison_results.csv", index=False)
print("\nSaved comparison table → model_comparison_results.csv")
print(df_results)
