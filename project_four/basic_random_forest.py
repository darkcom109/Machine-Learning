import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")   # non-GUI backend, only saves files
import matplotlib.pyplot as plt

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

# --- Decision Tree ---
tree_model = DecisionTreeClassifier(random_state=42).fit(X, y)

print("Decision Tree Prediction for [160g, smooth]:", tree_model.predict([[160, 0]]))
print("Decision Tree Prediction for [215g, bumpy]:", tree_model.predict([[215, 1]]))

# Save decision tree visualization
plt.figure(figsize=(6, 4))
plot_tree(tree_model, feature_names=["Weight", "Texture"], class_names=["Apple", "Orange"], filled=True)
plt.savefig("decision_tree.png")
print("Saved visualization as decision_tree.png")

# --- Random Forest ---
forest_model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

print("Random Forest Prediction for [160g, smooth]:", forest_model.predict([[160, 0]]))
print("Random Forest Prediction for [215g, bumpy]:", forest_model.predict([[215, 1]]))