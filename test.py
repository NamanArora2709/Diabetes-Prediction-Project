import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the cleaned dataset
file_path = "diabetes_dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File '{file_path}' not found. Please make sure it's in the same directory.")

df = pd.read_csv(file_path)
print("📄 Columns in dataset:", df.columns.tolist())

# Step 2: Set the correct target column name
TARGET_COLUMN = "Y"  # ✅ Your dataset uses 'Y' as the target

if TARGET_COLUMN not in df.columns:
    raise KeyError(f"❌ Target column '{TARGET_COLUMN}' not found in dataset. Please check the column names.")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# Step 5: Save the trained model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Model saved as 'diabetes_model.pkl' using pickle.")

# Step 6: Reload the model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded successfully from 'diabetes_model.pkl' using pickle.")

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation")
print("----------------------------")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R-squared (R²) Score     : {r2:.2f}")

# Step 9: Visualization
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Diabetes Progression")
plt.ylabel("Predicted Diabetes Progression")
plt.title("Actual vs Predicted Diabetes Progression")
plt.tight_layout()
plt.show()