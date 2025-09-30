import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Features: [weight, texture]
# Labels: 0 = apple, 1 = orange
X = np.array([
    [150, 0],
    [170, 0],
    [140, 0],
    [200, 1],
    [210, 1],
    [220, 1]
])
y = np.array([0, 0, 0, 1, 1, 1])

# Train Random Forest with 10 trees, more trees -> usually more stable
forest_model = RandomForestClassifier(n_estimators=10, random_state=42)
forest_model.fit(X, y)

# Predictions
print("Random Forest Prediction for [160g, smooth]:", forest_model.predict([[160, 0]]))  # Apple
print("Random Forest Prediction for [215g, bumpy]:", forest_model.predict([[215, 1]]))  # Orange
