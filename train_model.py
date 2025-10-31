import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib

# Load datasets
fake_df = pd.read_csv(r"C:\Users\STSSC\OneDrive\Desktop\FakeNewsDetection\Fake.csv", on_bad_lines='skip', encoding='utf-8')
true_df = pd.read_csv(r"C:\Users\STSSC\OneDrive\Desktop\FakeNewsDetection\True.csv", on_bad_lines='skip', encoding='utf-8')

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'TRUE'

# Combine and shuffle
data = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = data['text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
model = joblib.load("fake_news_model.joblib")
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
