## If X_test contains 4 features, you should separate them into individual columns in your DataFrame. Here's the corrected code:

# Assuming X_test is your feature matrix with shape (30, 4)
# And pred is your predictions array with shape (30,)

# Create a DataFrame with the features from X_test
final_df = pd.DataFrame(X_test, columns=['feature1', 'feature2', 'feature3', 'feature4'])

# Add the predictions as a new column
final_df['predicted'] = pred

## If you specifically wanted to have just two columns named 'actual_test' and 'predicted', you could do something like: 

# If you want to store X_test as a Series of arrays
final_df = pd.DataFrame({
    'actual_test': [row for row in X_test],  # Each row of X_test becomes an element
    'predicted': pred
})