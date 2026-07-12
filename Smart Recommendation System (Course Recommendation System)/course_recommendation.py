import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("courses.csv")

# Encode categorical data
interest_encoder = LabelEncoder()
skill_encoder = LabelEncoder()
course_encoder = LabelEncoder()

data["interest_encoded"] = interest_encoder.fit_transform(data["interest"])
data["skill_encoded"] = skill_encoder.fit_transform(data["skill_level"])
data["course_encoded"] = course_encoder.fit_transform(
    data["recommended_course"]
)

# Features and Target
X = data[["interest_encoded", "skill_encoded"]]
y = data["course_encoded"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Accuracy Evaluation
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

print("Smart Recommendation System")
print("-" * 40)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# User Input
interest = input("Enter Interest: ")
skill = input("Enter Skill Level (Beginner/Intermediate/Advanced): ")

input_data = pd.DataFrame({
    "interest_encoded": [
        interest_encoder.transform([interest])[0]
    ],
    "skill_encoded": [
        skill_encoder.transform([skill])[0]
    ]
})

prediction = model.predict(input_data)

recommended_course = course_encoder.inverse_transform(prediction)

print("\nRecommended Course:")
print(recommended_course[0])

# -----------------------------
# Course Popularity Bar Chart
# -----------------------------

course_counts = data["interest"].value_counts()

plt.figure(figsize=(12,6))
course_counts.plot(kind="bar")
plt.title("Course Category Popularity")
plt.xlabel("Course Category")
plt.ylabel("Number of Courses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Domain Distribution Pie Chart
# -----------------------------

domain_counts = data["interest"].str.split().str[0].value_counts()

plt.figure(figsize=(8,8))
plt.pie(
    domain_counts,
    labels=domain_counts.index,
    autopct="%1.1f%%"
)
plt.title("Distribution of Course Domains")
plt.show()
