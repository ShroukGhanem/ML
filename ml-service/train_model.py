from sklearn.linear_model import LinearRegression
import joblib

# ----------------------------
# 1. Simple training dataset
# ----------------------------
# We are modelling a simple relationship: y = 2x

X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# ----------------------------
# 2. Create model
# ----------------------------
model = LinearRegression()

# ----------------------------
# 3. Train model
# ----------------------------
model.fit(X, y)

# ----------------------------
# 4. Save trained model
# ----------------------------
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
