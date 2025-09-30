import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[150], [160], [170], [180], [190]])
y = np.array([50, 55, 65, 72, 80])

model = LinearRegression().fit(X, y)

# Makes 100 evenly spaced numbers between 140 and 200, turns the list into a column vector (2D array)
x_values = np.linspace(140, 200, 100).reshape(-1, 1)

# Takes each of those height and predicts a weight using the regression line
y_preds = model.predict(x_values)

# Plots the original dataset as red dots
plt.scatter(X, y, color="red", label="Training data")

# Plots the regression line
plt.plot(x_values, y_preds, color="blue", label="Fit line")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Linear Regression Example")
plt.legend()

plt.savefig("linear_regression.png")
print("Saved graph as linear_regression.png")

print("Predicted weight for 175cm:", model.predict([[175]])[0])