## Solution
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

# Create a list of text documents from the 'message' column
X_train_texts = X_train['message'].tolist()

# Initialize and fit the CountVectorizer on training data
vec = TfidfVectorizer()
X_train_count = vec.fit_transform(X_train_texts).toarray()

# Train the Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_count, y_train)

# Prepare test data
emails = [
    'Hey mohan, can we watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

# Transform test data using the same vectorizer
emails_count = vec.transform(emails).toarray()

# Make predictions
predictions = nb_model.predict(emails_count)
print(predictions)




# Check the shapes of your test data
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Make sure X_test is properly formatted before transformation
if isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series):
    X_test_texts = X_test['message'].tolist() if 'message' in X_test.columns else X_test.tolist()
else:
    X_test_texts = X_test

# Transform the correctly formatted test texts
X_test_count = vec.transform(X_test_texts).toarray()

# Verify the shapes after transformation
print(f"X_test_count shape: {X_test_count.shape}")
print(f"y_test shape: {y_test.shape}")

# Now score the model
score = nb_model.score(X_test_count, y_test)
print(f"Model accuracy: {score:.4f}")


## Solution to not vectorized data 
# Create a pipeline with vectorization and classification
clf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Fit the pipeline with raw text data
clf.fit(X_train, y_train)  # X_train should be raw text, not vectorized

# Score with raw text data
score = clf.score(X_test, y_test)  # X_test should also be raw text
print(f"Accuracy: {score}")



***## Solution to already vectorized data****
# If you want to use already vectorized data, skip the vectorizer in the pipeline
clf = MultinomialNB()
clf.fit(X_train_trans, y_train)

# Make sure X_test is also transformed the same way before scoring
X_test_trans = vec.transform(X_test_vec).toarray()  # Use your vectorizer 'vec'
score = clf.score(X_test_trans, y_test)
print(f"Accuracy: {score}")