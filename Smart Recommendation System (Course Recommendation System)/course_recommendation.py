import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("courses.csv")

# Label Encoding
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nSMART RECOMMENDATION SYSTEM")
print("-" * 40)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Display available options
print("\nAvailable Interests:")
for item in interest_encoder.classes_:
    print("-", item)

print("\nAvailable Skill Levels:")
for item in skill_encoder.classes_:
    print("-", item)

# User Input
interest = input("\nEnter Interest: ").strip()
skill = input(
    "Enter Skill Level (Beginner/Intermediate/Advanced): "
).strip()

# Validation
if interest not in interest_encoder.classes_:
    print("\nInvalid Interest Entered.")
    exit()

if skill not in skill_encoder.classes_:
    print("\nInvalid Skill Level Entered.")
    exit()

# Encode user input
input_data = pd.DataFrame({
    "interest_encoded": [
        interest_encoder.transform([interest])[0]
    ],
    "skill_encoded": [
        skill_encoder.transform([skill])[0]
    ]
})

# Prediction
prediction = model.predict(input_data)

recommended_course = course_encoder.inverse_transform(
    prediction
)

print("\nRecommended Course:")
print(recommended_course[0])

# Bar Chart
course_counts = data["interest"].value_counts()

plt.figure(figsize=(10, 5))
course_counts.plot(kind="bar")
plt.title("Course Category Popularity")
plt.xlabel("Interest")
plt.ylabel("Number of Courses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart
domain_counts = data["interest"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    domain_counts,
    labels=domain_counts.index,
    autopct="%1.1f%%"
)
plt.title("Distribution of Interests")
plt.show()


