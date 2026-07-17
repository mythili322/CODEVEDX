import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("courses.csv")

print("\nSMART RECOMMENDATION SYSTEM")
print("-" * 40)

# Display available interests
print("\nAvailable Interests:")
for interest in sorted(data["interest"].unique()):
    print("-", interest)

# Display available skill levels
print("\nAvailable Skill Levels:")
for skill in sorted(data["skill_level"].unique()):
    print("-", skill)

# User Input
interest = input("\nEnter Interest: ").strip()
skill = input(
    "Enter Skill Level (Beginner/Intermediate/Advanced): "
).strip()

# Validate input
if interest not in data["interest"].unique():
    print("\nInvalid Interest Entered!")
    exit()

if skill not in data["skill_level"].unique():
    print("\nInvalid Skill Level Entered!")
    exit()

# Get recommendations
recommendations = data[
    (data["interest"] == interest) &
    (data["skill_level"] == skill)
]["recommended_course"]

print("\nRecommended Courses:")
print("-" * 25)

for i, course in enumerate(recommendations, start=1):
    print(f"{i}. {course}")

# -----------------------------
# Course Category Popularity Bar Chart
# -----------------------------

interest_counts = data["interest"].value_counts()

plt.figure(figsize=(12, 6))
interest_counts.plot(kind="bar")
plt.title("Course Category Popularity")
plt.xlabel("Interest Domain")
plt.ylabel("Number of Courses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Interest Distribution Pie Chart
# -----------------------------

plt.figure(figsize=(10, 10))
plt.pie(
    interest_counts,
    labels=interest_counts.index,
    autopct="%1.1f%%"
)
plt.title("Distribution of Course Domains")
plt.tight_layout()
plt.show()


