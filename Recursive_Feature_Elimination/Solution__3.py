## 1. Use the coefficients directly ## 
from sklearn.linear_model import LogisticRegression
import numpy as np

# After fitting your model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Get feature importances from coefficients
feature_importances = np.abs(lr.coef_[0])  # Take absolute values
feature_names = X_train.columns  # or your feature names

# Create a sorted list
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

## 2. Use permutation importance ##
pythonfrom sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(lr, X_test, y_test, n_repeats=10, random_state=42)

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

## 3. For standardized features, use coefficients directly ##
pythonfrom sklearn.preprocessing import StandardScaler

# Standardize features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit logistic regression
lr = LogisticRegression()
lr.fit(X_scaled, y_train)

# Now coefficients are more directly comparable
feature_importances = np.abs(lr.coef_[0])