import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['study_hours', 'sleep_hours', 'attendance']]
y = data['marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Take user input
study = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
attendance = float(input("Enter attendance (%): "))

# Predict
prediction = model.predict([[study, sleep, attendance]])
print("Predicted Marks:", prediction[0])
