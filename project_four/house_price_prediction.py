import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")   # ensure we can save plots without opening a window
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Housing.csv")

# Encode categorical yes/no columns into 0/1
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# One-hot encode furnishingstatus (furnished / semi-furnished / unfurnished)
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# Features (all except price)
X = df.drop("price", axis=1)
y = df["price"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Coefficients
print("Intercept (b):", model.intercept_)
print("Coefficients (w):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef}")

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  
# red dashed line = perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("house_price_predictions.png")
print("Saved graph as house_price_predictions.png")
