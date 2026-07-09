import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("news.csv")

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vectorized, y_train)

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("AI Fake News Detection Tool")
print("==============================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

plt.bar(["Accuracy"], [accuracy * 100])
plt.ylabel("Accuracy (%)")
plt.title("Fake News Detection Model Accuracy")
plt.show()

while True:
    news = input("\nEnter News Text (type exit to quit): ")

    if news.lower() == "exit":
        print("Application Closed")
        break

    news_vector = vectorizer.transform([news])
    result = model.predict(news_vector)[0]

    if result == "REAL":
        print("\nPrediction: REAL NEWS")
    else:
        print("\nPrediction: FAKE NEWS")
