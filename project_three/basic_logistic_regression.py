from sklearn.model_selection import train_test_split # Dataset splitter
from sklearn.linear_model import LogisticRegression # A simple model for the data

# Logistic Regression is similar to linear regression, however it provides only an output
# out of two possible options provided, in this example, 1 or 0.
# This is essentially category prediction vs numerical prediction

# The entire machine learning pipeline is:
# 1) Have data (numbers + labels)
# 2) Split into train and test
# 3) Pick a model (Logistic Regression)
# 4) Train it with fit()
# 5) Use it with .predict()

# Each input must be placed in an array because scikit-learn expects 2D arrays
# The dataset is then mapped via
# [1] -> 0, [2] -> 0, [3] -> 0, [10] -> 1, [11] -> 1, [12] -> 1
X = [[1], [2], [3], [10], [11], [12]] # Features/inputs
y = [0, 0, 0, 1, 1, 1] # Labels/outputs, 0 = small, 1 = big

# Split the data into two groups:
# - Training set, examples the model learns from
# - Test set, examples the model has never seen (to check if it generalises)
# {test_size=0.5} means half for training and half for testing
# {random_state} just makes the split reproducible (always the same)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Tries to find a boundary line that seperates 0s and 1s (the cutoff point)
model = LogisticRegression()

# .fit() trains the model and tries to figure out the rule
# Logistic regression uses linear algebra + calculus
model.fit(X_train, y_train)

print("Prediction for 2:", model.predict([[2]])) # should be 0 (small)
print("Prediction for 6:", model.predict([[6.01]])) # should be 1 (big)



