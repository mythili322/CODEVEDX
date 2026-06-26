import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Display dataset information
print("\nDataset Information")
print("-" * 40)
print(data.info())

print("\nMissing Values")
print("-" * 40)
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

# Features and Labels
X = data["text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train Model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n")
print("=" * 50)
print("        AI-Based Fake News Detection Tool")
print("=" * 50)
print(f"Model Accuracy : {accuracy * 100:.2f}%")

# Save Model
joblib.dump(model, "model.pkl")
print("\nTrained model saved as model.pkl")

# User Input
print("\nEnter a News Headline")
news = input(">> ")

# Prediction
prediction = model.predict([news])[0]

# Confidence Score
probability = model.predict_proba([news]).max() * 100

print("\nPrediction :", prediction)
print(f"Confidence : {probability:.2f}%")

print("\nThank you for using the AI-Based Fake News Detection Tool!")
