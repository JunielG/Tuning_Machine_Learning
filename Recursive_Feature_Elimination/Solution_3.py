## Solution 1: Ensure Your Pipeline Has a Compatible Estimator ##
Make sure the final estimator in your pipeline has either coef_ (for linear models) or feature_importances_ (for tree-based models):

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Example with RandomForestClassifier (has feature_importances_)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Or with LogisticRegression (has coef_)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

## Solution 2: Specify the Importance Getter Explicitly ##
If your estimator has feature importance but RFECV can't find it automatically, specify it explicitly:

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# For tree-based models
rfecv = RFECV(
    estimator=pipeline,
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy',
    min_features_to_select=1,
    importance_getter='named_steps.classifier.feature_importances_',  # Specify path
    n_jobs=-1,
    verbose=1
)

# For linear models
rfecv = RFECV(
    estimator=pipeline,
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy',
    min_features_to_select=1,
    importance_getter=lambda x: x.named_steps.classifier.coef_[0],  # For binary classification
    n_jobs=-1,
    verbose=1
)

## Solution 3: Use RFE Instead of RFECV (If You Don't Need CV) ##
If you don't specifically need cross-validation for feature selection:

from sklearn.feature_selection import RFE
# Fit the pipeline first
pipeline.fit(X, y)

# Then use RFE
rfe = RFE(
    estimator=pipeline,
    n_features_to_select=10,  # Specify number of features
    step=1,
    verbose=1
)
rfe.fit(X, y)

## Solution 4: Complete Working Example ##
Here's a complete example that should work:

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline with compatible estimator
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Create RFECV object
rfecv = RFECV(
    estimator=pipeline,
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy',
    min_features_to_select=1,
    n_jobs=-1,
    verbose=1
)

# Fit RFECV
print("Fitting RFECV...")
rfecv.fit(X, y)

# Access results
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {rfecv.support_}")
print(f"Feature ranking: {rfecv.ranking_}")
