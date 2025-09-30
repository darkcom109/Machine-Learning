import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-GUI backend/saves to a file
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a toy dataset
# Features: [hours studied, classes attended]
X = np.array([
    [1, 5],
    [2, 7],
    [3, 8],
    [4, 8],
    [5, 10],
    [6, 9],
    [7, 10],
    [8, 9],
    [9, 10],
    [10, 10]
])

y = np.array([35, 45, 50, 55, 65, 70, 75, 80, 88, 95])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

print("Intercept (b):", model.intercept_)
print("Coefficients (w)", model.coef_)
print("Interpretation: score = (hours_studied * w1) + (classes_attended * w2) + b")

hours = np.linspace(0, 10, 100)
attendance_fixed = 9
X_line = np.c_[hours, np.full(hours.shape, attendance_fixed)]
y_line = model.predict(X_line)

plt.scatter(X[:,0], y, color="red", label="Data (hours vs score)")
plt.plot(hours, y_line, color="blue", label=f"Regression line (attendance={attendance_fixed})")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.title("Linear Regression: Predicting Student Scores")
plt.savefig("student_scores.png")
print("Saved plot as student_scores.png")