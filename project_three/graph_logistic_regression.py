import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-GUI backend/saves to a file
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [10], [11], [12]]) # Inputs
y = np.array([0, 0, 0, 1, 1, 1]) # Outputs

# Creates a logistic regression model and .fit() trains it
# Looks at numbers and labels. learns a boundary where 0s stop and 1s start
model = LogisticRegression().fit(X, y)

# Create 200 evenly spaced numbers between 0 and 13, turn it into a 2D column
# For each number, return the probability of belonging to class 0 and class 1
x_values = np.linspace(0, 13, 200).reshape(-1, 1)
y_probs = model.predict_proba(x_values)[:, 1]

# Data drawn as red dots
plt.scatter(X, y, color="red", label="Training data (0 = small, 1 = big)")

# Plots the smooth sigmoid showing how the probability of being "big" changes
plt.plot(x_values, y_probs, color="blue", label="Sigmoid curve")

# Draw a dashed horizontal line at y = 0.5
plt.axhline(0.5, color="gray", linestyle="--", label="0.5 cutoff")

plt.xlabel("Number (X)")
plt.ylabel("Probability of being class 1 (big)")
plt.title("Logistic Regression on Toy Dataset")

# Saves as an image
plt.legend()
plt.savefig("logistic_curve.png")
print("Saved graph as logistic_curve.png")
