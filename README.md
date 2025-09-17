# SGD-Regressor-for-Multivariate-Linear-Regression


## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: A SANJAY
RegisterNumber: 25016505
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
# -----------------------------
# Step 1: Create synthetic dataset
# -----------------------------
# Features: [Area (sq ft), Number of Rooms, Location Index]
X = np.array([
    [1200, 3, 1],
    [1500, 4, 2],
    [800, 2, 1],
    [2000, 5, 3],
    [1700, 4, 2],
    [1000, 2, 1],
    [2200, 5, 3],
    [1300, 3, 2]
])
# Targets: [Price (in lakhs), Number of Occupants]
y = np.array([
    [50, 4],
    [65, 5],
    [35, 3],
    [90, 7],
    [70, 6],
    [40, 3],
    [100, 8],
    [55, 4]
])
# -----------------------------
# Step 2: Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 3: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# -----------------------------
# Step 4: Train SGD Regressor
# -----------------------------
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant')
multi_regressor = MultiOutputRegressor(sgd)
multi_regressor.fit(X_train, y_train)

# -----------------------------
# Step 5: Prediction
# -----------------------------
y_pred = multi_regressor.predict(X_test)
# -----------------------------
# Step 6: Evaluation
# -----------------------------
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

```
R2 Score: 0.9346628530848737

MSE: 3.6312615925289293
```
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted.round(2)}")


```
## Output:

<img width="439" height="149" alt="image" src="https://github.com/user-attachments/assets/99cb2d7c-35fe-4388-910b-eaac9111e750" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
