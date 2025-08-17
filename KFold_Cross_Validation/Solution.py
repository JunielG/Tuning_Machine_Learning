from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import numpy as np

# Load the digits dataset
digits = load_digits()

# Define a scoring function if not already defined
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Setup cross-validation
folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

# Make sure digits.data and digits.target are NumPy arrays
X = np.array(digits.data)
y = np.array(digits.target)

for train_index, test_index in folds.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), 
                                    X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), 
                              X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), 
                             X_train, X_test, y_train, y_test))

# Print results
print("Logistic Regression scores:", scores_logistic)
print("SVM scores:", scores_svm)
print("Random Forest scores:", scores_rf)

print("Average Logistic Regression score:", sum(scores_logistic)/len(scores_logistic))
print("Average SVM score:", sum(scores_svm)/len(scores_svm))
print("Average Random Forest score:", sum(scores_rf)/len(scores_rf))