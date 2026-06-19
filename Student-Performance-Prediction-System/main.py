import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data.csv")

X = data[['attendance', 'study_hours', 'marks']]
y = data['performance']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred) * 100)

# User Input
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours: "))
marks = float(input("Enter Current Marks: "))

prediction = model.predict([[attendance, study_hours, marks]])

print("Predicted Performance:", prediction[0])
