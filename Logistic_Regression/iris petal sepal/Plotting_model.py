# Option 1: Plot predictions against one specific feature (e.g., the first feature)
plt.figure(figsize=(10,8))
plt.scatter(pred, X_test[:, 0])  # Using the first feature (index 0)
plt.xlabel('Predicted Values')
plt.ylabel('Feature 1 Values')
plt.show()

# Option 2: Create multiple scatter plots, one for each feature
plt.figure(figsize=(15, 10))
for i in range(4):  # Assuming X_test has 4 features
    plt.subplot(2, 2, i+1)  # Create a 2x2 grid of subplots
    plt.scatter(pred, X_test[:, i])
    plt.xlabel('Predicted Values')
    plt.ylabel(f'Feature {i+1} Values')
    plt.title(f'Predictions vs Feature {i+1}')
plt.tight_layout()
plt.show()