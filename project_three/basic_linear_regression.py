# Linear regression
# Involves collecting data points and forming a line of best fit for predictions
# y = wx + b, y = prediction, w = gradient, x = input, b = intercept

import numpy as np
from sklearn.linear_model import LinearRegression

# Heights in cm, weights in kg
X = np.array([[150], [160], [170], [180], [190]])
y = np.array([50, 55, 65, 72, 80])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

print("Slope (w):", model.coef_[0]) # Gradient
print("Intercept (b):", model.intercept_) # Base weight when height = 0

# Make a prediction
height = [[175]]
predicted_weight = model.predict(height)
print(f"Predicted weight for {height[0][0]} cm:", predicted_weight[0])